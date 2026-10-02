"""Perception Agent.

Extracts visible chart components from the input image. Delegates the actual
vision work to `ChartVisionTool`; this agent's job is just to invoke the
tool and pass the JSON through as its final answer, where it gets validated
against the PerceptionOutput schema.
"""

from __future__ import annotations

from crewai import Agent, LLM, Task

from chart_agents.config import settings
from chart_agents.schemas import PerceptionOutput
from chart_agents.tools.offline_vision import OfflineChartVisionTool
from chart_agents.tools.vision import ChartVisionTool


def build_perception_agent() -> Agent:
    return Agent(
        role="Chart Perception Specialist",
        goal=(
            "Look at the chart image and produce a complete, structured inventory of every "
            "visible chart element (title, axes, ticks, legend, bars, lines, points) along "
            "with estimated values and approximate bounding boxes."
        ),
        backstory=(
            "You are a meticulous computer-vision expert who has annotated thousands of "
            "scientific plots. You never invent elements that are not visible, and when "
            "an exact reading is impossible you provide a clearly-marked estimate and note "
            "the uncertainty."
        ),
        tools=[OfflineChartVisionTool() if settings.perception_backend != "vlm" else ChartVisionTool()],
        llm=LLM(model=settings.llm_model, temperature=settings.llm_temperature),
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


def build_perception_task(agent: Agent, image_path: str) -> Task:
    description = (
        f"Use the `chart_vision` tool with image_path='{image_path}'. The tool returns "
        f"a JSON object describing the chart. Return that JSON object verbatim as your "
        f"final answer. If the JSON contains an `error` field, propagate it. Do NOT add "
        f"markdown fences, do NOT add commentary, and do NOT modify the values."
    )
    return Task(
        description=description,
        expected_output=(
            "A JSON object with keys chart_type, title, x_axis_label, y_axis_label, "
            "x_tick_labels, y_tick_labels, legend, components, notes."
        ),
        agent=agent,
        output_pydantic=PerceptionOutput,
    )
