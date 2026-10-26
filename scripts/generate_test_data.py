"""Regenerates the synthetic PNG test images in data/test_data (answer scripts, diagrams, snap images).
Run: python scripts/generate_test_data.py   (needs Pillow)"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "data" / "test_data"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def answer_sheet(src: Path, dst: Path, header: str):
    text = src.read_text(encoding="utf-8").replace("\n", "\n\n")
    img = Image.new("RGB", (1000, 1200), (252, 250, 240))
    d = ImageDraw.Draw(img)
    for y in range(110, 1180, 38):
        d.line([(60, y), (940, y)], fill=(190, 205, 235), width=1)
    d.line([(90, 0), (90, 1200)], fill=(230, 150, 150), width=2)
    d.text((110, 30), header, font=font(26), fill=(20, 20, 120))
    f, y = font(24), 80
    for para in text.split("\n\n"):
        for ln in wrap(d, para, f, 800):
            d.text((110, y), ln, font=f, fill=(25, 25, 90))
            y += 38
    img.save(dst)


def box(d, xy, text, kind):
    x, y = xy
    f = font(18)
    if kind in ("start", "end"):
        d.ellipse([x - 70, y - 25, x + 70, y + 25], outline="black", width=3)
    elif kind == "decision":
        d.polygon([(x, y - 45), (x + 80, y), (x, y + 45), (x - 80, y)], outline="black", width=3)
    elif kind == "io":
        d.polygon([(x - 60, y + 25), (x - 80, y - 25), (x + 80, y - 25), (x + 60, y + 25)], outline="black", width=3)
    else:
        hw = max(80, int(d.textlength(text, font=f) / 2) + 14)
        d.rectangle([x - hw, y - 25, x + hw, y + 25], outline="black", width=3)
    w = d.textlength(text, font=f)
    d.text((x - w / 2, y - 10), text, font=f, fill="black")


def arrow(d, a, b, label="", head=True):
    d.line([a, b], fill="black", width=3)
    if head:
        ax, ay = b
        dx = (b[0] > a[0]) - (b[0] < a[0])
        dy = (b[1] > a[1]) - (b[1] < a[1])
        if dx == 0:
            pts = [(ax, ay), (ax - 6, ay - 10 * dy), (ax + 6, ay - 10 * dy)]
        else:
            pts = [(ax, ay), (ax - 10 * dx, ay - 6), (ax - 10 * dx, ay + 6)]
        d.polygon(pts, fill="black")
    if label:
        d.text(((a[0] + b[0]) / 2 + 6, (a[1] + b[1]) / 2 - 20), label, font=font(16), fill="black")


def flowchart(dst: Path, broken: bool):
    img = Image.new("RGB", (800, 700), "white")
    d = ImageDraw.Draw(img)
    box(d, (400, 50), "Start", "start")
    box(d, (400, 150), "Read n", "io")
    box(d, (400, 270), "n > 0 ?", "decision")
    box(d, (190, 400), "Print positive", "process")
    arrow(d, (400, 75), (400, 125)); arrow(d, (400, 175), (400, 225))
    arrow(d, (320, 270), (190, 270), head=False); arrow(d, (190, 270), (190, 375), "Yes")
    if broken:
        box(d, (600, 400), "Increment counter", "process")  # orphan, no End, one branch only
    else:
        box(d, (610, 400), "Print not positive", "process")
        box(d, (400, 560), "End", "end")
        arrow(d, (480, 270), (610, 270), head=False); arrow(d, (610, 270), (610, 375), "No")
        arrow(d, (190, 425), (190, 560), head=False); arrow(d, (190, 560), (330, 560))
        arrow(d, (610, 425), (610, 560), head=False); arrow(d, (610, 560), (470, 560))
    img.save(dst)


def er(dst: Path):
    img = Image.new("RGB", (900, 450), "white")
    d = ImageDraw.Draw(img)
    f = font(18)
    d.rectangle([80, 180, 230, 240], outline="black", width=3); d.text((125, 200), "Student", font=f, fill="black")
    d.rectangle([670, 180, 820, 240], outline="black", width=3); d.text((725, 200), "Course", font=f, fill="black")
    d.polygon([(450, 150), (540, 210), (450, 270), (360, 210)], outline="black", width=3); d.text((415, 200), "Enrolls", font=f, fill="black")
    d.line([(230, 210), (360, 210)], fill="black", width=3); d.text((280, 185), "N", font=f, fill="black")
    d.line([(540, 210), (670, 210)], fill="black", width=3); d.text((600, 185), "M", font=f, fill="black")
    for (cx, cy, t, a) in [(155, 60, "RegNo (PK)", (155, 180)), (60, 330, "Name", (130, 240)), (745, 60, "CourseCode (PK)", (745, 180))]:
        d.ellipse([cx - 70, cy - 28, cx + 70, cy + 28], outline="black", width=3)
        d.text((cx - d.textlength(t, font=f) / 2, cy - 10), t, font=f, fill="black")
        d.line([(cx, cy + 28 if cy < 200 else cy - 28), a], fill="black", width=2)
    img.save(dst)


def textbook_page(dst: Path):
    img = Image.new("RGB", (900, 800), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Chapter 3: The Stack", font=font(34), fill="black")
    body = ("A stack is a linear data structure in which insertion and deletion happen at one end called the top. "
            "The last element pushed is the first element popped (LIFO). Push places an item on top; pop removes it. "
            "Both operations take constant time, O(1).")
    y = 110
    for ln in wrap(d, body, font(22), 780):
        d.text((60, y), ln, font=font(22), fill="black"); y += 34
    for i, (v, c) in enumerate([("5", 650), ("12", 600), ("30", 550)]):
        d.rectangle([380, c, 520, c + 50], outline="black", width=3)
        d.text((440, c + 12), v, font=font(24), fill="black")
    d.text((545, 555), "<- top", font=font(22), fill="black")
    d.text((330, 700), "PUSH (down)  /  POP (up)", font=font(22), fill="black")
    img.save(dst)


def rc_circuit(dst: Path):
    img = Image.new("RGB", (800, 500), "white")
    d = ImageDraw.Draw(img)
    d.line([(100, 100), (300, 100)], fill="black", width=4)
    pts = [(300, 100)]
    for i in range(6):
        pts.append((300 + 20 * (i + 1), 70 if i % 2 == 0 else 130))
    pts.append((440, 100)); d.line(pts, fill="black", width=4)
    d.line([(440, 100), (650, 100), (650, 220)], fill="black", width=4)
    d.line([(620, 220), (680, 220)], fill="black", width=5); d.line([(620, 245), (680, 245)], fill="black", width=5)
    d.line([(650, 245), (650, 400), (100, 400), (100, 250)], fill="black", width=4)
    d.ellipse([70, 170, 130, 250], outline="black", width=4); d.text((95, 195), "V", font=font(26), fill="black")
    d.line([(100, 170), (100, 100)], fill="black", width=4)
    d.text((340, 30), "R = 1 kOhm", font=font(22), fill="black"); d.text((690, 215), "C = 10 uF", font=font(22), fill="black")
    d.text((120, 440), "Series RC circuit, V = 5 V. Find the time constant.", font=font(22), fill="black")
    img.save(dst)


def parabola(dst: Path):
    img = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(img)
    d.line([(400, 40), (400, 560)], fill="black", width=2); d.line([(40, 500), (760, 500)], fill="black", width=2)
    pts = [(400 + x * 30, 500 - (x * x) * 18) for x in [i / 4 for i in range(-14, 15)]]
    d.line(pts, fill="blue", width=3)
    d.text((410, 45), "y", font=font(22), fill="black"); d.text((740, 510), "x", font=font(22), fill="black")
    d.text((60, 60), "Graph of y = x^2. Where is the minimum value?", font=font(24), fill="black")
    img.save(dst)


if __name__ == "__main__":
    A = ROOT / "answer_scripts"
    for name in ("Q3_good", "Q3_partial", "Q3_poor"):
        answer_sheet(A / f"{name}.txt", A / f"{name}.png", "Q3. Stack")
    answer_sheet(A / "Q5_good.txt", A / "Q5_good.png", "Q5. Binary search")
    flowchart(ROOT / "diagrams" / "flowchart_correct.png", broken=False)
    flowchart(ROOT / "diagrams" / "flowchart_broken.png", broken=True)
    er(ROOT / "diagrams" / "er_diagram.png")
    textbook_page(ROOT / "snap" / "textbook_stack_page.png")
    rc_circuit(ROOT / "snap" / "circuit_rc.png")
    parabola(ROOT / "snap" / "graph_parabola.png")
    print("generated")
