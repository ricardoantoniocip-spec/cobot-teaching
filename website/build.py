"""Build the static website from src/ into this folder.

Change the company name in one place (BRAND below), then run:  python build.py
Pages are plain HTML afterwards: open index.html in a browser, or host the folder anywhere.
"""
from pathlib import Path

BRAND = "Robotonic"
DOMAIN = ""  # e.g. "robotonic.example" once a domain is chosen; used only for the canonical link

HERE = Path(__file__).parent
SRC = HERE / "src"

PAGES = [
    # file, nav key, <title>, meta description
    ("index.html", "home", "{BRAND} | Teach a cobot by showing it",
     "{BRAND} turns a video of an operator doing a task into a verified cobot program. URScript for Universal Robots cobots today."),
    ("how-it-works.html", "how", "How it works | {BRAND}",
     "From one video to a cobot program: record, track, label, generate, verify."),
    ("service.html", "service", "New task in a week | {BRAND}",
     "A fixed-price service that sets up a new cobot task in one week, for integrators and high-mix factories."),
    ("about.html", "about", "About | {BRAND}",
     "Why {BRAND} exists, who is behind it and where it is going."),
    ("404.html", "none", "Page not found | {BRAND}", "This page does not exist."),
    ("contact.html", "contact", "Tell us your task | {BRAND}",
     "Describe a task you want a cobot to learn. We reply with what is possible and what it would take."),
]

NAV = [("how", "how-it-works.html", "How it works"),
       ("service", "service.html", "Service"),
       ("about", "about.html", "About")]


def nav_html(active: str) -> str:
    items = []
    for key, href, label in NAV:
        cur = ' aria-current="page"' if key == active else ""
        items.append(f'<li><a href="{href}"{cur}>{label}</a></li>')
    cur = ' aria-current="page"' if active == "contact" else ""
    items.append(f'<li><a class="btn btn-signal" href="contact.html"{cur}>Tell us your task</a></li>')
    return "\n          ".join(items)


def main() -> None:
    layout = (SRC / "_layout.html").read_text(encoding="utf-8")
    for fname, key, title, desc in PAGES:
        body = (SRC / fname).read_text(encoding="utf-8")
        canonical = f'<link rel="canonical" href="https://{DOMAIN}/{"" if fname == "index.html" else fname}">' if DOMAIN else ""
        page = (layout
                .replace("{{BODY}}", body)
                .replace("{{NAV}}", nav_html(key))
                .replace("{{TITLE}}", title)
                .replace("{{DESCRIPTION}}", desc)
                .replace("{{CANONICAL}}", canonical)
                .replace("{BRAND}", BRAND))
        (HERE / fname).write_text(page, encoding="utf-8")
        print("wrote", fname)


if __name__ == "__main__":
    main()
