# Background

## Problem Statement

Electric Vehicle (EV) Supply Equipment (EVSE) — the hardware that meters and
delivers charging power to EVs — is being connected to the grid in ever-growing
numbers. Modern chargers are networked devices that communicate over protocols
such as the Open Charge Point Protocol (OCPP), making them attractive and
high-impact targets for cyberattackers.

A successful attack against an EVSE can disrupt charging availability, skew
billing and energy metering, destabilize the local distribution grid, or act as
a pivot into utility backend systems. Traditional intrusion-detection systems
(IDS) rely on network-packet inspection at the protocol layer, which requires
deep visibility into charging infrastructure that operators may not have in
place.

## Research Motivation

This project investigates a **non-invasive, content-agnostic** alternative:
detecting ongoing cyberattacks using **only the electrical power-consumption
telemetry** that EVSE already records as part of normal operations (shunt
voltage, bus voltage, current, and instantaneous power draw).

Because every computation a compromised charger performs — packet floods,
connection churn, reboot cycles, unexpected state transitions — ultimately
manifests as a measurable change in electrical load, the physical power
footprint is a promising, hypothesis-free signal for attack detection.

## Why Temporal Validation Matters

Power-telemetry datasets are **time series**. A naïve random
train/test split distributes temporally adjacent measurements between the
training and test sets. Because adjacent samples are highly correlated, the
model effectively sees "the test answer" during training, producing
artificially perfect scores (~99.9% accuracy) that do not generalize to unseen
future conditions.

This project treats **temporal / chronological validation** as a first-class
concern, comparing a strict chronological (out-of-distribution) split against a
leakage-aware stratified-by-time-block split, and demonstrating that honest,
transferable performance estimates are substantially lower than the naive
figure.

## Key References

- EVSE Cybersecurity Dataset — Dataset B, *PowerCombined* telemetry
  (OCPP-enabled charger, ~6 days of 1-second sensor samples).
- Open Charge Point Protocol (OCPP) — charging-communication standard.
- Feature: Isolation Forest anomaly scoring for unsupervised signal extraction.
- Explainability: SHAP (SHapley Additive exPlanations), TreeExplainer.

See [`references/`](../references/) for the source material.
