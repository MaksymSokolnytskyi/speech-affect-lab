# Data: sources, licences, verified numbers

No audio is stored in this repository. The download scripts put it in `data/raw/` and verify
every file against its published SHA-256.

## CREMA-D

| Field | Value | Source |
|---|---|---|
| Repository | https://github.com/CheyneyComputerScience/CREMA-D (mirror: https://gitlab.com/cs-cooper-lab/crema-d-mirror) | repository README |
| Paper | Cao H. et al. *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Trans. Affective Computing 5(4):377–390, 2014. doi:10.1109/TAFFC.2014.2336244 | repository README |
| Licence | Open Database License 1.0 (database); Database Contents License 1.0 (contents) | `LICENSE.txt` in the repository |
| Clips | 7,442 | README; checked by `test_counts_match_official_description` |
| Actors | 91: 48 male, 43 female, aged 20–74 | README; checked by `test_sex_split_matches_official_description` |
| Sentences | 12 fixed, semantically neutral English sentences | README |
| Emotions | 6: anger, disgust, fear, happy, neutral, sad | README |
| Intensity | low, medium, high, unspecified (`XX`); LO/MD/HI exist only for sentence IEO | README + file-name count |
| Class balance | 1,271 clips per emotion, neutral 1,087 | file-name count |
| Human accuracy (audio only, intended emotion) | 40.9% | Cao et al., 2014 |
| Listener ratings | 2,443 raters, >7 ratings for 95% of clips; audio-only votes in `processedResults/summaryTable.csv`, column `VoiceVote` | README |

**Authors' request.** The README asks users who obtain the data from GitHub to fill in a
short Google form so the authors know their user community.

**Integrity.** The repository stores audio in git-lfs. We clone only the pointers
(`GIT_LFS_SKIP_SMUDGE=1`); each pointer carries the official SHA-256. Every WAV is then
downloaded individually from `media.githubusercontent.com` and checked against that hash
(`speech_affect.data.cremad`).

**Labels.** The intended emotion (from the file name) is the target. The audio-only listener
vote is kept in the manifest for analysis (`voice_vote`, `voice_vote_emotion`; a tie is
stored as e.g. `N:H` and leaves `voice_vote_emotion` empty).

**What the manifest shows.**
- All 7,442 hashes match the git-lfs pointers; all files are 16 kHz mono.
- Duration: mean 2.54 s, range 1.27–5.0 s; 5.26 hours in total. No corrupted clips.
- 644 clips end in a tie of the audio-only listener vote.
- On the remaining 6,798 clips, the listener majority names the intended emotion in only
  **45.6%** of cases, and defaults to "neutral" when unsure (74% of *sad* clips are heard as
  neutral). The intended labels are therefore noisy, and many portrayals are subtle.

## MUSAN (noise for the telephone experiments)

| Field | Value | Source |
|---|---|---|
| Official page | https://www.openslr.org/17 (11.1 GB archive) | OpenSLR |
| What we download | unofficial Hugging Face mirror https://huggingface.co/datasets/FluidInference/musan, `noise/` only (~585 MB); per-file SHA-256 from the mirror's git-lfs metadata | — |
| Licence | CC BY 4.0; per-file licences in `noise/*/LICENSE` | OpenSLR; Snyder, Chen, Povey, 2015, arXiv:1510.08484 |
| `noise/free-sound` | public domain | `noise/free-sound/LICENSE` |
| `noise/sound-bible` | CC BY 3.0, author and link per file | `noise/sound-bible/LICENSE` |
| Files | 774 WAV | count after download |

Noise files are split deterministically (CRC32 of the file name) into a training pool (~70%)
and a test pool (~30%), so augmentation and the test conditions never share a noise
recording.

Caveat: the mirror is unofficial. The hashes guarantee the files are not corrupted in
transit, not that the mirror is identical to the OpenSLR archive.
