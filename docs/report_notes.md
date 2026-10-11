# Report notes: motivation, context and statistics

Working notes for the RailSight report, gathered 10 Oct 2026. Every number has a source; items
marked ⚠ need checking against the primary document (a parliamentary reply, NCRB table or ERA
report) before they go into the final report.

---

## 1. The problem in numbers

### India: official "train accidents" vs actual deaths on the railway

| Measure | Value | Source |
|---|---|---|
| Consequential train accidents, 2023-24 (collisions, derailments, fires, LC accidents) | **40** (down from 473 in 2000-01) | Ministry of Railways, Lok Sabha replies ([AU1387](https://eparlib.sansad.in/bitstream/123456789/2984233/1/AU1387_hYnOEw.pdf), [AIR, Dec 2024](https://www.newsonair.gov.in/rail-accidents-decline-sharply-over-last-decade-railways-minister)) |
| Breakdown 2023-24 | 6 collisions, 25 derailments, 0 level-crossing, 9 fires | [Dataful compilation of Lok Sabha data](https://dataful.in/datasets/1210/) |
| Deaths in consequential accidents, 2014-15 to 2023-24 (10 years) | 748 | same Lok Sabha replies |
| **Railway accident cases recorded by NCRB, 2023** | **24,678** | NCRB ADSI 2023 ([The Wire](https://thewire.in/government/21803-people-killed-in-24678-railway-accidents-across-india-in-2023-ncrb-data), [ETV Bharat](https://www.etvbharat.com/en/!bharat/national-crime-records-bureau-2023-data-21803-killed-in-rail-accidents-enn25093006012)) |
| **Deaths in railway accidents, NCRB 2023** | **21,803** | same |
| Share caused by "fall from trains or collision with people on track" | **74.9 % of cases, 72.8 % of deaths (15,878)** | same |
| Same category, 2019 | 76.3 % of cases, 74.5 % of deaths (18,339 of 24,619) | [Deccan Herald](https://www.deccanherald.com/amp/story/india%2Frailway-accidents-killed-24619-in-2019-says-ncrb-data-882970.html) |
| NCRB 2022 | 23,139 cases, ~20,792 deaths ⚠ (compilation, not the PDF) | [Dataful](https://dataful.in/datasets/20194) |

**Key point for the report:** the Ministry's safety statistics count *train accidents*. People and
animals struck on the track are almost entirely outside that count, yet they make up roughly three
quarters of the ~22,000 railway deaths a year. This is exactly the gap a forward-looking
obstacle detector addresses.

⚠ The NCRB category merges *falls from trains* with *people struck on the track*; the split is
in the ADSI tables and should be checked before quoting a number for strikes alone.

### Animals

| Measure | Value | Source |
|---|---|---|
| Cattle run-over incidents | 27,057 (2019-20), 19,960 (2020-21), 28,727 (2021-22), 26,180 (2022-23 to 23 Jan) | Rajya Sabha written reply, Railway Minister ([railanalysis](https://news.railanalysis.com/indian-railways-takes-various-measures-to-prevent-cattle-run-over-incidents/)) ⚠ |
| Cattle run-over incidents (calendar years) | 26,181 (2021), 27,489 (2022) | [Rajya Sabha, 31 Mar 2023](https://rsdebate.nic.in/bitstream/123456789/739679/1/PQ_259_31032023_U3503_p374_p375.pdf) |
| Elephants killed by trains | 186 (2009-10 to 2020-21); Assam 62, West Bengal 57, Odisha 27 | MoEFCC via [Drishti IAS](https://www.drishtiias.com/daily-updates/daily-news-analysis/death-of-186-elephants-in-10-years) ⚠ |
| Elephants killed by trains, last 5 years | 79 (2020-21 to 2024-25) | Environment Ministry to Parliament, Aug 2025 ([The Week](https://www.theweek.in/wire-updates/national/2025/08/18/del73-lsq-env-elephants.html)) |

So roughly **70+ cattle strikes per day** on Indian Railways. Each one also delays trains
(the replies list trains losing punctuality) and damages locomotives.

### Europe / UK (for international context)

| Measure | Value | Source |
|---|---|---|
| Level crossings' share of accidental rail fatalities (30+ countries, 2021) | 32 % of fatalities, 27 % of significant accidents | UIC safety report ([ILCAD press release](https://uic.org/com/IMG/pdf/ilcad_press_release_en_new.pdf)) |
| Third parties' share of rail fatalities (trespassers, LC users) | 96 % of fatalities | same |
| Share of public fatality risk on GB mainline from trespass | 82 % | UK Office of Rail and Road ([Rail safety 2023-24](https://dataportal.orr.gov.uk/media/hizn3q3n/rail-safety-2023-24.pdf)) |

⚠ For an EU-wide figure, use ERA's *Report on Railway Safety and Interoperability in the EU*
(Common Safety Indicators). Not retrieved yet.

---

## 2. Why the driver alone is not enough

- **Stopping distance:** an express train needs roughly **700 m to 1 km** to stop from ~110 km/h
  (Indian railway engineers quoted in [DT Next / PressReader, May 2024](https://www.pressreader.com/india/dt-next/20240504/281663965087606);
  [railwaysignallingconcepts](https://www.railwaysignallingconcepts.in/railway-train-braking-concepts/)).
  ⚠ No official RDSO figure found; quote as an approximate range.
- Our own figure (`reports/figures/stopping_distance.png`): at 100 km/h with 2 s reaction and
  1 m/s² braking, ~440 m; a 1.7 m person is then ~6 px tall on a full-HD, 60° camera.
- Human factors: long monotonous shifts, night, fog, curves, tunnel exits, divided attention
  (signals, cab displays).
- Even when a stop is impossible, earlier braking cuts impact energy (∝ v²) and an earlier horn
  gives people and animals time to clear the track.
- Analogy: cars have drivers, yet forward-collision warning / AEB is standard because it reacts
  consistently.

---

## 3. What exists and what does not

| System | What it does | What it does **not** do |
|---|---|---|
| **Kavach** (Indian ATP, SIL-4) | Prevents signals passed at danger, overspeed, and train-to-train collisions; cab signalling (useful in fog); auto-whistle at level crossings | Does **not** detect people, cattle, elephants or debris on the track |
| Kavach deployment | v4.0 commissioned on ~738 route km by Dec 2025 (Palwal-Mathura-Nagda 633 km, Howrah-Barddhaman 105 km); trackside work underway on 15,512 rkm | ([Swarajya](https://swarajyamag.com/amp/story/infrastructure%2Frailways-fast-tracks-kavach-deployment-738-km-commissioned-on-delhimumbai-delhihowrah-routes-work-advancing-on-15000-km-more), [Hitavada, RTI](https://www.thehitavada.com/Encyc/2025/11/26/automatic-train-protection-system-kavach-commissioned-on-654-km-of-routes-till-sep-2025-rti.html)) |
| **Intrusion Detection System / "Gajraj"** (elephants) | Fibre-optic distributed acoustic sensing detects elephants near the track in mapped corridors and alerts the controller / loco pilot. NFR since Dec 2022 on 11 corridors; expansion to ~700 km (₹181 crore) | Fixed corridors only; tuned to elephants; not a general obstacle detector ([YourStory/Mongabay](https://yourstory.com/socialstory/2024/11/ai-system-prevent-train-elephant-collisions-conservationists-questions), [Daily Excelsior](https://www.dailyexcelsior.com/news/archive/railways-to-implement-ai-based-surveillance-system-on-700-km-tracks-to-prevent-elephant-deaths), [Sensonic](https://www.sensonic.com/en/start/elephants-get-rail-protection-with-fiber-optics-and-ai--3259)) |
| Driverless metros (GoA4) | Rely on segregated track: fences, tunnels, viaducts, platform screen doors, contact obstacle-detection bars | Not applicable to open mainline track |
| **SMART** (EU Shift2Rail, 2016-19) | Research prototype fusing thermal, night-vision, stereo and laser for obstacle detection up to 1,000 m | Research demonstrator, not deployed ([CORDIS 730836](https://cordis.europa.eu/project/id/730836)) |
| **OSDaR23** (DZSF, DB Netz / Digitale Schiene Deutschland, FusionSystems) | Released to support **automatic train operation** development: 45 sequences, multi-camera + IR + lidar + radar, 1,534 annotated multi-sensor frames, 204,091 annotations, 20 classes | Its existence shows that open-track obstacle perception is an **open research problem** ([arXiv 2305.03001](https://arxiv.org/pdf/2305.03001), [FID move](https://data.fid-move.de/dataset/osdar23)) |

---

## 4. What RailSight builds on and what it adds

**Uses (existing work):**
- Data: RailSem19 (AIT, CVPRW 2019), two Roboflow railway-obstacle datasets (MIT, CC BY 4.0),
  possibly OSDaR23 for lidar-validated ranging; Pexels clips for qualitative demos.
- Models: SegFormer (NVIDIA, Cityscapes-pretrained), DeepLabV3 (torchvision, COCO-pretrained),
  YOLO11 and RT-DETR (Ultralytics, COCO-pretrained), ByteTrack.

**Adds (our contributions):**
- Audited, leak-free merge of the obstacle datasets (class mapping, duplicate grouping by
  perceptual hash, stratified re-split), which also shows the published splits leaked.
- Ego-track corridor from segmentation, with **rail-gauge metric ranging** (no depth sensor).
- Class-agnostic unknown-object check, with an evaluation showing why the low-confidence
  heuristic hurts.
- Interpretable risk engine: braking physics, in-path prediction over the arrival window,
  hysteresis.
- Quantitative evaluation of every stage, plus an honest "where physics limits camera
  detection" analysis.

---

## 5. Viva pitch (draft)

> RailSight is a driver-assistance and research prototype for the open-track obstacle problem
> that existing train protection (Kavach) does not cover. For every hazard it answers two
> questions: is it in the train's path, and can the train still stop? It also quantifies where
> camera-based detection helps (lower speeds, trams, yards, level crossings) and where braking
> physics limits any onboard sensor (high-speed mainline).

## Still to collect
- ⚠ ERA EU-wide Common Safety Indicators (trespasser / level-crossing fatality shares).
- ⚠ NCRB ADSI table separating "fall from train" and "struck on track".
- Official RDSO braking distances by train type (if publicly available).

---

## 6. Key results and findings to report (all numbers in README "Results", generated from reports/)

1. **Segmentation:** SegFormer-B2 mIoU 0.696 / track IoU 0.912 (RailSem19 test). DeepLabV3-R50 ties on
   track IoU (0.914) but its corridor geometry is worse (~20% higher ranging deviation): pixel
   overlap is not the metric that matters for ranging.
2. **Corridor:** found on 99.9% of frames with a real track; segmentation adds 3.8 / 7.3 / 13.5%
   ranging deviation at 25 / 50 / 100 m.
3. **Ranging without calibration:** gauge vs human-height distance ratio 0.995 for 708 people at track
   level (no scale bias; spread ±31%, mostly the reference). People on platforms: 0.77, as the
   flat-ground geometry predicts.
4. **Detection domain gap (important):** detectors fine-tuned on Roboflow images score 0.67–0.71 mAP50
   on the Roboflow test but only 0.06–0.08 on real cab-view frames (worse than un-tuned COCO, 0.19).
   Adding ~16k training boxes derived from RailSem19 masks (v2) gives 0.43 on cab-view frames (person AP 0.52)
   without losing Roboflow accuracy (0.716). Lesson: test on the deployment viewpoint.
   **Resolution:** running v2 at 960 px lifts cab-view mAP50 to 0.462 (small-object recall 0.19 → 0.26);
   fine-tuning at 960 px adds a little more (0.476) but no small-object recall. Distant objects remain
   the main limitation. The 960 px model is the default and costs <1 FPS (full pipeline ~28 FPS).
5. **Unknown objects:** class-only rule precision 0.58 / recall 0.48 / 4.9% false alarms; adding
   low-confidence pixels quadruples false alarms. Temporal persistence and a low-visibility guard
   further cut false alarms on real video.
6. **Speed:** full pipeline ~28 FPS on an idle RTX 5060 Laptop (17.7-28 across sessions; profiling-driven
   optimisation gave +41% on SegFormer-B2 back-to-back, predictions unchanged).
7. **Negative results worth stating:** rail-pair gauge measurement (helps at merged switches, hurts
   elsewhere); motion-based vehicle mask (rails ahead are as static as the dashboard); Ultralytics
   `time=` budget silently stretches the LR schedule; YOLO11m vs YOLO11s differences are within the
   noise of a 270-image test set.
