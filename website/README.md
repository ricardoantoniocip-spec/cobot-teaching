# Robotonic website

Static, multi-page company website. No backend, no tracking, no cookies.

Pages: `index.html` (home), `how-it-works.html`, `service.html` ("new task in a week"), `about.html`, `contact.html` ("tell us your task").

## View it
Run `python -m http.server` in this folder and go to http://localhost:8000. (Opening `index.html` directly also works, but browsers then block the local fonts.)

## Change something
- Text: edit the page in `src/`, then run `python build.py`. Do not edit the top-level `.html` files by hand; the build overwrites them.
- Company name: change `BRAND` in `build.py` and run `python build.py`. Logo text, titles and copy all follow.
- Contact email: set `CONTACT_EMAIL` at the top of `assets/main.js`. While it is empty, the form writes the message for the visitor to copy.
- Fonts are in `assets/fonts/` (Archivo, Instrument Sans, JetBrains Mono, all SIL Open Font License), served from the site so no visitor data goes to a font service.
- `404.html` is the not-found page; most hosts pick it up automatically.
- Look: colours, fonts and spacing are tokens at the top of `assets/style.css` (light and dark).

## Assets
- `assets/video/robotonic-ad.mp4`: 42 s prototype ad with voice-over.
- `assets/video/video-to-ur-program.mp4`: 52 s walkthrough, video to URScript to URSim and the SO-101 twin.
- `assets/img/`: stills from those videos and the take 1 tracking.

## Before going live
- Not published anywhere yet. Choose a domain and host (GitHub Pages, Netlify or Cloudflare Pages all serve this folder as is).
- Universal Robots is named in plain text only, with a "not affiliated" note in the footer. Do not add their logo without permission.
- Everything shown is a prototype in simulation; keep the status claims in `how-it-works.html` current.
