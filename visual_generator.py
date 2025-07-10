import os
import re
import uuid
from graphviz import Digraph  #type: ignore
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # <- ✅ Add this line
import matplotlib.pyplot as plt



def needs_visual_aid(query: str) -> bool:
    """
   Checks if the query specifically requests a flowchart or diagram.
    """
    keywords = [
        "flowchart", "flow chart", "chart", "graph",
        "visual aid", "visualization", "visual representation", 
        "process", "cycle", "architecture", "steps",
        "block diagram", "chart", "bar chart", "pie chart", "line chart",
        "line graph", "data visualization", "data chart", "data plot",
        "data graph", "data diagram", "data flow", "data representation",
        "histogram", "scatter plot", "graph", "visualization", "visual aid",
        "visual representation", "workflow", "decision tree", "mind map",
        "sequence diagram", "state machine", "UML diagram", "Gantt chart",
        "org chart", "network diagram", "data flow diagram", "ERD",
        "entity relationship diagram", "class diagram",
        "use case diagram", "activity diagram", "swimlane diagram",
        "timeline", "infographic", "schematic", "circuit diagram",
        "architecture diagram", "system diagram", "process flow",
    ]
    return any(word in query.lower() for word in keywords)


def extract_steps_from_text(text: str) -> list:
    """
    Extracts step headers and conditional branches for flowcharts.
    """
    cutoff_keywords = ["note", "notes", "tip", "explanation", "summary", "additional", "reference"]
    for keyword in cutoff_keywords:
        idx = text.lower().find(keyword)
        if idx != -1:
            text = text[:idx]
            break

    steps = []
    decision_lines = re.findall(r"(?:^|\n)([^\n]*?\u2192\s*Yes:.*?\u2192\s*No:.*?)", text)
    steps.extend([line.strip() for line in decision_lines])

    for line in decision_lines:
        text = text.replace(line, '')

    numbered = re.findall(r"(?:^|\n)\d+\.\s*(.+?)(?=\n|$)", text)
    steps.extend([step.strip() for step in numbered if len(step.strip()) < 70])

    if not numbered:
        bullets = re.findall(r"(?:^|\n)[\u2022\*-]\s*(.+?)(?=\n|$)", text)
        steps.extend([b.strip() for b in bullets if len(b.strip()) < 70])

    return steps


def generate_flowchart(title: str, steps: list) -> str:
    dot = Digraph()
    dot.attr(rankdir="TB", bgcolor="white")
    dot.attr("node", fontname="Arial", fontsize="11", style="filled")

    clean_steps = [step.strip() for step in steps if step.strip()]

    if clean_steps and re.match(r"(?i)(flowchart|diagram|generated)", clean_steps[0]):
        clean_steps.pop(0)

    node_id = 0
    last_node = None

    dot.node("start", "Start", color="greenyellow", shape="ellipse")
    last_node = "start"

    for step in clean_steps:
        if "→" in step and ":" in step:
            match = re.match(r"(.+?)→\s*Yes:\s*(.+?)→\s*No:\s*(.+)", step)
            if match:
                question, yes_branch, no_branch = match.groups()
                q_id = f"q{node_id}"
                y_id = f"y{node_id}"
                n_id = f"n{node_id}"
                dot.node(q_id, question.strip(), shape="diamond", color="lightyellow")
                dot.node(y_id, yes_branch.strip(), shape="box", color="lightblue2")
                dot.node(n_id, no_branch.strip(), shape="box", color="lightblue2")
                dot.edge(last_node, q_id)
                dot.edge(q_id, y_id, label="Yes")
                dot.edge(q_id, n_id, label="No")
                last_node = n_id
                node_id += 1
                continue

        cur_id = f"n{node_id}"
        dot.node(cur_id, step, shape="box", color="lightblue2")
        dot.edge(last_node, cur_id)
        last_node = cur_id
        node_id += 1

    dot.node("end", "End", color="orangered", shape="ellipse")
    dot.edge(last_node, "end")

    output_path = f"static/flowchart_{uuid.uuid4().hex[:8]}"
    dot.render(output_path, format="png", cleanup=True)
    return output_path + ".png"


def generate_chart(chart_type: str, labels: list, values: list, title: str = "Generated Chart") -> str:
    plt.clf()
    fig, ax = plt.subplots()

    if chart_type == "bar":
        ax.bar(labels, values, color="skyblue")
    elif chart_type == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%")
    elif chart_type == "line":
        ax.plot(labels, values, marker="o")
    elif chart_type == "histogram":
        ax.hist(values, bins=5, color="purple", alpha=0.7)
    elif chart_type == "scatter":
        ax.scatter(labels, values, color="green")
    else:
        return None

    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"static/chart_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(filename)
    return filename


def cleanup_old_visuals(folder="static", minutes=10):
    now = datetime.now()
    for filename in os.listdir(folder):
        if filename.endswith(".png") and (
            filename.startswith("flowchart") or filename.startswith("chart")
        ):
            path = os.path.join(folder, filename)
            modified = datetime.fromtimestamp(os.path.getmtime(path))
            if now - modified > timedelta(minutes=minutes):
                os.remove(path)
