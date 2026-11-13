# Demo Capture (product/capture)

A phone page to film a demo, check it, and send it to training.

Link: https://claude.ai/artifact/UrYZPFExMBD7ZPXqPshPqw (private to Ricardo; open it in Safari on the iPhone, signed in to Claude).

```
record on the phone -> scan (on the phone + Claude looks at 6 frames) -> send -> training queue -> so101_sim from-video
```

## What the scan checks

| Check | How | Result if it fails |
|---|---|---|
| File size under 20 MB | file size | Fix (cannot be sent) |
| Length 4-20 s | video duration | Fix under 4 s, Check over 20 s |
| Picture size | video resolution | Check |
| Light, sharpness, camera kept still | 6 frames, measured on the phone | Fix if very dark, else Check |
| Whole arm in view, hand visible, a pick and a place shown | Claude looks at the 6 frames | Fix |
| Thumb and index grip, same flat surface, enough light | Claude looks at the 6 frames | Check |

The scan is a filter, not the real tracker. The real check is `so101_sim from-video` after the video arrives.

## After sending

Videos are stored with the page (20 MB each). Each sent take is a row in the page's `takes` list:
`asset` (video id), `arm`, `seconds`, `scan` (ok / warn / bad), `checks`, `status`.
Claude fetches the video, runs `python -m so101_sim from-video`, and sets `status` to
`tracked`, `episode` or `failed` with a short `result` text, which shows on the phone.
This step is done by Claude on request for now ("process the new takes"). It is not automatic yet.

## Tested (12 Nov 2026)

- In a phone-sized Chromium with a stand-in for Claude: the full flow on Ricardo's first video, light
  and dark theme, no errors. A steady video scores 2.6 on camera movement, a shaken copy 8.8 (limit 6).
- Not tested: a real iPhone, the real Claude answer, a real upload. The thresholds for light and
  sharpness are first guesses.

The page source is `demo-capture.html`. Publish changes to the same link.
