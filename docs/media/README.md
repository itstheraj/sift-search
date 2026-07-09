# Demo media

`search-window.png`, `settings.png`, and `dolphin-menu.png` are used by the Demo
section of the root `README.md`. A `demo.webm` can go here too.

## Screenshots

Capture a single window rather than the whole screen, so nothing else on your
desktop ends up in the shot:

```bash
spectacle --region --background --nonotify --output docs/media/search-window.png
```

Check the file before committing it. A screenshot picks up whatever is behind
the window, including other applications.

## Screencast

Plasma's Spectacle records a region to webm:

```bash
spectacle --record region --output docs/media/demo.webm
```

Keep it under about twenty seconds and around 1280 wide. If it comes out large,
shrink it:

```bash
ffmpeg -i docs/media/demo.webm -vf scale=1280:-2 -c:v libvpx-vp9 -crf 34 -b:v 0 -an docs/media/demo-small.webm
```

GitHub will not play a webm that is referenced by a repository path. To get an
inline player, drag the file into a GitHub issue comment, copy the
`user-images.githubusercontent.com` URL it generates, and paste that URL into
the README on its own line. Do not submit the issue.

## Shot list

A demo works best if it proves the thing a file manager cannot already do.

1. Open the search window with an empty index visible, then run `sift reindex`
   on a folder of images. Let the progress bar run. It is quick.
2. Search `something you can eat`. Food comes back. Nothing in those filenames
   says food.
3. Search `the night sky`. The aurora photographs come back.
4. Search `underwater`. The reef photographs come back.
5. Hit Alt+Space and run the same query through KRunner, to show it is wired
   into the desktop rather than being a standalone window.
6. Right click a folder in Dolphin and pick "Search here with Sift".
7. Finish on a video result opening at its timestamp in mpv.

Steps 2 through 4 are the whole pitch. Spend the most time there.

## Test corpus

The images used for the reference demo are 100 public domain and CC0 files from
Wikimedia Commons. They are not committed to this repository. Any folder of
images works.
