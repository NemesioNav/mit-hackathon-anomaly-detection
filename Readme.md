# Anomaly Detection for Thermal Drone Footage

**Track**: Fine-Tuning of Models

## 1. Motivation

Manual review of drone footage—especially in time-critical operations like search and rescue—is slow, error-prone, and costly. Anomaly detection models have shown excellent results in industrial quality assurance but are rarely applied to humanitarian contexts.

This project explores how existing anomaly detection models can be adapted for high-impact use cases, such as locating missing persons using thermal drone imagery or identifying defects in industrial inspections. The goal is to demonstrate the value of transferring AI from industrial to humanitarian domains.

## 2. Features

### A. Search & Rescue – Thermal Drone Imagery

Prototype capabilities:
- Classifies thermal drone frames as **normal** (no person) or **abnormal** (potential human presence).
- Ranks abnormal images to help responders focus on the most critical frames.
- Supports image upload and simulated real-time analysis of drone footage.

**Optional extensions**:
- GPS integration or location overlays.
- Simulated alert systems (e.g., dashboard, SMS, email).

### B. Functional Requirements

- Minimal user interface (web or CLI) to upload and process images.
- Prioritization logic (e.g., anomaly score sorting).
- Use of transfer learning: fine-tune industrial models (e.g., trained on MVTec AD) for thermal rescue data.
- Visual feedback encouraged (e.g., bounding boxes, saliency maps, anomaly heatmaps).

## 3. Dataset

- **Source**: Black Forest Mountain Rescue Team (Global MIT Hackathon)
- **Format**: 226 thermal images taken with a DJI Matrice 300 (M300) + Zenmuse H20T camera
- **Conditions**: 90° camera angle, 80m altitude, 11°C ambient temperature

**Breakdown**:
- 166 training images (normal)
- 60 test images:  
  - 30 normal  
  - 30 abnormal (with thermal signatures indicating human presence)

**Note**: No public dataset exists for this domain. Supplement or simulate using public drone datasets, synthetic data, or open thermal imagery.

## 4. Evaluation Criteria

| **Criterion**                    | **Goal**                                                                 |
|----------------------------------|--------------------------------------------------------------------------|
| Anomaly Detection Effectiveness  | Minimize false positives/negatives in classifying abnormal images        |
| Domain Adaptation & Generalization | Adapt pre-trained models effectively to thermal rescue domain             |
| User-Centered Design & Usability | Ensure accessibility for non-technical users (e.g., rescue teams)         |
| Creativity & Innovation          | Add novel features: alerts, automation, UI enhancements                   |
| Clarity & Documentation          | Provide clear code, comments, and explanations of the methodology         |

## 5. Impact

### Humanitarian Impact

In emergencies, speed matters. AI-assisted drone footage analysis can drastically reduce the time to locate missing persons, increasing survival chances and operational efficiency. An open-source solution like this could support emergency responders globally.

### Industrial Impact

Anomaly detection in manufacturing is a massive challenge. Automating it boosts safety, quality, and efficiency. This project bridges industrial methods and real-world applications, proving AI’s flexibility across domains.
