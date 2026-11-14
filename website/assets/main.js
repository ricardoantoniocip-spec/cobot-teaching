/* Robotonic website script. No tracking, no cookies, no backend. */

/* Where "Tell us your task" messages go. Leave empty until the company address exists:
   the form then shows the message for the visitor to copy. */
const CONTACT_EMAIL = "";

(function () {
  // mobile navigation
  const toggle = document.querySelector(".nav-toggle");
  const links = document.getElementById("nav-links");
  if (toggle && links) {
    toggle.addEventListener("click", () => {
      const open = links.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  if (toggle && links) {
    const close = () => { links.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); };
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    links.addEventListener("click", (e) => { if (e.target.closest("a")) close(); });
  }

  // light / dark switch, remembered on this device
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const root = document.documentElement;
      const now = root.getAttribute("data-theme") ||
        (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = now === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* private mode: not remembered */ }
    });
  }

  // footer year
  document.querySelectorAll("[data-year]").forEach((el) => { el.textContent = new Date().getFullYear(); });

  // hero pendant: step through the demo, highlighting the matching program lines
  const pendant = document.querySelector("[data-pendant]");
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (pendant && !reduce) {
    const chips = Array.from(pendant.querySelectorAll(".chip"));
    const lines = Array.from(pendant.querySelectorAll(".ln[data-step]"));
    const readout = pendant.querySelector("[data-readout]");
    const reads = [
      "t = <b>0.0 s</b>", "t = <b>3.1 s</b>", "t = <b>5.8 s</b>", "t = <b>8.5 s</b>", "t = <b>9.5 s</b>"
    ];
    let i = 0;
    const show = (k) => {
      chips.forEach((c, n) => c.classList.toggle("on", n === k));
      lines.forEach((l) => l.classList.toggle("on", Number(l.dataset.step) === k));
      if (readout) readout.innerHTML = reads[k];
    };
    setInterval(() => { i = (i + 1) % chips.length; show(i); }, 2200);
  }

  // contact form: compose the message in the browser, nothing is stored or sent by the site
  const form = document.getElementById("task-form");
  if (form) {
    const result = document.getElementById("task-result");
    const out = document.getElementById("task-text");
    const mail = document.getElementById("task-mail");
    const copy = document.getElementById("task-copy");
    const note = document.getElementById("task-note");
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      if (!form.reportValidity()) return;
      const v = (id) => (document.getElementById(id).value || "").trim();
      const text = [
        "New task request",
        "",
        "Name: " + v("f-name"),
        "Company: " + v("f-company"),
        "Email: " + v("f-email"),
        "Country: " + v("f-country"),
        "We are: " + v("f-role"),
        "Robot: " + v("f-robot"),
        "How often the task changes: " + v("f-change"),
        "",
        "The task:",
        v("f-task")
      ].join("\n");
      out.textContent = text;
      result.hidden = false;
      if (CONTACT_EMAIL) {
        mail.hidden = false;
        mail.href = "mailto:" + CONTACT_EMAIL + "?subject=" + encodeURIComponent("Task request: " + v("f-company")) +
          "&body=" + encodeURIComponent(text);
        note.textContent = "Your message is ready. Open it in your email app, or copy it and send it to " + CONTACT_EMAIL + ".";
      } else {
        mail.hidden = true;
        note.textContent = "Your message is ready. Copy it and send it to us by email or LinkedIn.";
      }
      result.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "nearest" });
    });
    copy.addEventListener("click", () => {
      const done = () => { copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy message"), 1800); };
      const fallback = () => {
        const r = document.createRange(); r.selectNodeContents(out);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        copy.textContent = "Selected: press copy";
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(out.textContent).then(done, fallback);
      } else { fallback(); }
    });
  }
})();
