"""Pydantic schemas for every stage of the chart reasoning pipeline.

The naming mirrors the current chart ontology used by the pipeline:

    group    = axis-side label, such as an x-axis bar group or y-axis bucket
    category = legend-side label, such as a colored legend/category name

    e_i = (type, label, group, category, value, bbox)  -> ChartComponent
    Chart DSL                                          -> ChartDSL
    q   = (task, target_groups, target_categories, operation) -> ParsedQuestion
    E   = {(group, category, value, visual_ref)}       -> EvidenceSet / EvidenceItem
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Perception Agent output
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Approximate location of a chart element in image coordinates.

    Coordinates may be normalised (0..1) or in pixels; downstream code does
    not assume a particular convention, but each PerceptionOutput should use
    one consistently.
    """

    x: float = Field(..., description="Top-left x")
    y: float = Field(..., description="Top-left y")
    w: float = Field(..., description="Width")
    h: float = Field(..., description="Height")


class ChartComponent(BaseModel):
    """A single detected chart element. e_i = (type, label, group, category, value, bbox)."""

    type: str = Field(
        ...,
        description="Element type: bar | line_segment | point | legend_entry | axis_label | tick | other",
    )
    label: Optional[str] = Field(None, description="Free-text label associated with the element")
    value: Optional[float] = Field(None, description="Estimated numerical value, if any")
    group: Optional[str] = Field(
        None,
        description="Axis-side label copied exactly from the chart, such as an x-axis bar group or y-axis bucket.",
    )
    category: Optional[str] = Field(
        None,
        description="Legend-side label copied exactly from the chart, such as a color/legend/category name.",
    )
    bbox: Optional[BoundingBox] = None


class PerceptionOutput(BaseModel):
    """Output of the Perception Agent."""

    chart_type: str = Field(
        ..., description="bar_chart | line_chart | scatter_plot | pie_chart | area_chart | other"
    )
    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    x_tick_labels: List[str] = Field(default_factory=list)
    y_tick_labels: List[str] = Field(default_factory=list)
    legend: List[str] = Field(default_factory=list)
    components: List[ChartComponent] = Field(default_factory=list)
    notes: Optional[str] = Field(
        None, description="Free-form notes from the perceiver (e.g. ambiguity, occlusion)"
    )


# ---------------------------------------------------------------------------
# Representation Agent output
# ---------------------------------------------------------------------------


class ChartDataPoint(BaseModel):
    """One cell of the chart's data table.

    Each point ties a (group, category) pair to a numeric value, with an
    optional pointer back to the visual mark in PerceptionOutput.components
    that produced it.

    This is a flat list-of-records instead of a nested dict because OpenAI's
    strict structured-output mode does not allow objects with arbitrary keys
    (which is what `Dict[str, X]` compiles to in JSON schema).
    """

    group: str = Field(
        ...,
        description="Axis-side label, such as an x-axis bar group or y-axis bucket.",
    )
    category: str = Field(
        ...,
        description="Legend-side label, such as a color/legend/category name.",
    )
    value: float
    visual_mark_index: Optional[int] = Field(
        None,
        description="0-based index into PerceptionOutput.components for the bar/point/segment "
        "that produced this value. Null if no clean match was found.",
    )


class ChartDSL(BaseModel):
    """Structured DSL representation of the chart.

    `data` is a flat list of ChartDataPoints; look up a value by filtering on
    (group, category). The links back to visual marks live on each point's
    `visual_mark_index`.
    """

    type: str
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    groups: List[str] = Field(
        default_factory=list,
        description="Axis-side labels, such as x-axis bar groups or y-axis buckets.",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Legend-side labels, such as color/legend/category names.",
    )
    data: List[ChartDataPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def deduplicate_data_points(self) -> "ChartDSL":
        """Keep one point per (group, category), preferring the largest value.

        Perception can occasionally emit overlapping or ambiguous bars with the
        same logical label pair. Downstream reasoning treats duplicate rows as
        real repeated cells, so normalize them here before any solver sees them.
        """

        deduped: dict[tuple[str, str], ChartDataPoint] = {}
        order: list[tuple[str, str]] = []
        for point in self.data:
            key = (point.group, point.category)
            if key not in deduped:
                deduped[key] = point
                order.append(key)
                continue
            if point.value > deduped[key].value:
                deduped[key] = point
        self.data = [deduped[key] for key in order]
        return self


# ---------------------------------------------------------------------------
# Question Parser Agent output
# ---------------------------------------------------------------------------


TaskKind = Literal[
    "value_lookup",
    "max",
    "min",
    "comparison",
    "trend",
    "interval_change",
    "counting",
    "multi_condition_filter",
    "other",
]


class ParsedQuestion(BaseModel):
    """Operation-level representation of the user's question.

    q = (task, target_groups, target_categories, operation)
    """

    task: TaskKind
    target_groups: List[str] = Field(
        default_factory=list,
        description="Subset of axis-side groups to consider, or empty list to mean 'all'.",
    )
    target_categories: List[str] = Field(
        default_factory=list,
        description="Subset of legend-side categories to consider, or empty list to mean 'all'.",
    )
    operation: Optional[str] = Field(
        None,
        description="Concise symbolic description, e.g. 'max(A - B) over all groups'.",
    )
    raw_question: str


# ---------------------------------------------------------------------------
# Evidence Agent output
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """(group, category, value, visual_ref). Values must come from the ChartDSL."""

    group: str = Field(
        ...,
        description="Axis-side label copied exactly from ChartDSL.data[*].group.",
    )
    category: str = Field(
        ...,
        description="Legend-side label copied exactly from ChartDSL.data[*].category.",
    )
    value: float
    visual_ref: Optional[str] = Field(
        None,
        description="Reference to the originating visual mark, e.g. 'components[3]'.",
    )


class EvidenceSet(BaseModel):
    items: List[EvidenceItem]
    rationale: str = Field(..., description="One-sentence justification for the selection.")


# ---------------------------------------------------------------------------
# Reasoning Agent output
# ---------------------------------------------------------------------------


class IntermediateValue(BaseModel):
    """A single named intermediate computation."""

    name: str
    value: float


class FinalAnswer(BaseModel):
    answer: str = Field(
        ..., description="Concise final answer suitable for benchmark grading."
    )
    explanation: str = Field(
        ..., description="Grounded explanation that references groups/categories from the evidence."
    )
    intermediate_values: List[IntermediateValue] = Field(
        default_factory=list,
        description="Intermediate computations the answer depends on.",
    )


# ---------------------------------------------------------------------------
# Critic Agent output
# ---------------------------------------------------------------------------


class CriticVerdict(BaseModel):
    consistent: bool
    issues: List[str] = Field(default_factory=list)
    suggested_revisit: Literal["evidence", "representation", "perception", "none"] = "none"


# ---------------------------------------------------------------------------
# Aggregate result
# ---------------------------------------------------------------------------


class PipelineResult(BaseModel):
    perception: PerceptionOutput
    chart_dsl: ChartDSL
    parsed_question: ParsedQuestion
    evidence: EvidenceSet
    answer: FinalAnswer
    critic: CriticVerdict
