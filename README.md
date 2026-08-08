# Patterson Simulator - Technical Guide

This document provides a technical overview of the Patterson Simulator, an interactive web application designed for educational purposes. The tool visualizes the relationship between a 2D crystal structure, its corresponding diffraction pattern, and its Patterson map (vector space).

## How to Start

To start the program, you can go [here](https://nitad54448.github.io/Patterson/Patterson.html).

## Table of Contents

* [Introduction & Core Concept](#introduction--core-concept)
* [The Patterson Function Explained](#the-patterson-function-explained)
* [Quick Start Guide](#quick-start-guide)
* [The User Interface](#the-user-interface)
    * [Controls Panel](#controls-panel)
    * [Results Area (The Four Plots)](#results-area-the-four-plots)
* [Algorithms & Calculations](#algorithms--calculations)
    * [1. Crystal Structure → Patterson Map](#1-crystal-structure--patterson-map)
    * [2. Crystal Structure → Diffraction Pattern](#2-crystal-structure--diffraction-pattern)
    * [3. Diffraction Pattern → Patterson Map](#3-diffraction-pattern--patterson-map)
    * [Interactive Vector Highlighting](#interactive-vector-highlighting)
* [About This Tool](#about-this-tool)

---

<section id="introduction--core-concept">

## Introduction & Core Concept

The Patterson Simulator is an interactive tool that demonstrates the fundamental principles of X-ray crystallography. It allows you to build a simple 2D crystal structure by placing atoms in a unit cell and instantly see the consequences in both reciprocal space (the diffraction pattern) and vector space (the Patterson map).

The core teaching concept is to illustrate two equivalent ways of generating the Patterson map:
1.  **Directly:** By calculating all interatomic vectors (atom-to-atom vectors) from the real-space structure.
2.  **Indirectly:** By taking the Fourier Transform of the simulated diffraction intensities, which is the method used to solve real-world structures.

The tool's main purpose is to help users understand *what* a Patterson peak represents by interactively linking the vector peaks back to the specific atoms that create them.

</section>

---

<section id="the-patterson-function-explained">

## The Patterson Function Explained 🗺️

A Patterson map, $P(u,v)$, is a **vector map** of the crystal structure. While the crystal structure shows *atom positions* $(x,y)$, the Patterson map shows *interatomic vectors* $(u,v)$.

For every pair of atoms in the unit cell (atom `i` and atom `j`), a peak is generated in the Patterson map at the position:
$$(u,v) = (x_j - x_i, y_j - y_i)$$

The **height (weight)** of this Patterson peak is proportional to the product of the scattering factors of the two atoms:
$$\text{Weight} \propto f_i \times f_j$$

This means that a vector between two heavy atoms will produce a very strong peak, while a vector between two light atoms will be weak. The map also includes a large "origin peak" at (0,0) from all the "self-vectors" (e.g., atom `i` to atom `i`), with a weight of $\sum f_i^2$.

Crucially, the Patterson function can be calculated *directly from experimental data* (the diffraction intensities, $|F_{hkl}|^2$) without knowing the structure, using a Fourier transform:
$$P(u,v) = \mathcal{F} ( |F_{hkl}|^2 )$$

This tool visualizes both methods simultaneously.

</section>

---

<section id="quick-start-guide">

## Quick Start Guide

1.  **Select Atom:** In the `Select Atom Type` panel, choose "Type A" or "Type B".
2.  **Add Atoms:** Click anywhere on the **`Crystal Structure (Real Space)`** canvas (top-left) to add an atom of the selected type at that $(x,y)$ coordinate.
3.  **Observe:** As you add atoms, all three other plots—`Patterson Map`, `Diffraction Pattern`, and `FT of Intensities`—will update automatically.
4.  **Explore Vectors:** Move your mouse over one of the colored peaks in the **`Patterson Map (Vector Space)`** canvas (top-right).
5.  **See the Link:** Observe the `Crystal Structure` canvas. The two atoms that generate that specific vector will be highlighted (one pink, one purple), and an arrow will be drawn between them. The `Vector Details` box will show the exact coordinates.
6.  **Adjust Parameters:**
    * Use the `Scattering Factors` sliders to change the "weight" of atoms A and B and see how the Patterson peak heights and Diffraction intensities change.
    * Use the `Diffraction Controls` sliders to change the resolution (`Range (h,k)`) of the diffraction pattern.

</section>

---

<section id="the-user-interface">

## The User Interface

The application is split into a `controls-panel` on the left and a `results-area` on the right.

### Controls Panel

* **Instructions:** Basic steps for using the simulator.
* **Select Atom Type to Add:** Choose between two different atom types, A (blue) or B (red), to add to the crystal structure.
* **Scattering Factors:** Sliders to control the scattering factor (number of electrons) for Atom A ($f_A$) and Atom B ($f_B$). This directly affects the peak weights in the Patterson map and the intensities in the diffraction pattern.
* **Diffraction Controls:**
    * `Range (h,k)`: Sets the $h$ and $k$ limits for calculating the diffraction pattern (e.g., -10 to +10).
    * `Spot Size Multiplier`: Adjusts the visual size of the diffraction spots for clarity.
    * `Show h,k Labels`: Toggles the visibility of the $(h,k)$ indices on the diffraction pattern.
* **Vector Details:** A text box that displays the coordinates of the "from" and "to" atoms and the resulting $(u,v)$ vector when you hover over a Patterson peak.
* **Atom List:** A list of all atoms you have added, showing their type and $(x,y)$ coordinates. You can manually edit the coordinates here or remove individual atoms.
* **Clear All Atoms:** A button to reset the crystal structure.

### Results Area (The Four Plots)

This is the main visualization area, composed of four linked canvases:

1.  **Crystal Structure (Real Space):**
    * **What it is:** The 2D unit cell. This is your "input" canvas.
    * **Interaction:** Click to add atoms. When you hover over the Patterson map, this plot will show the highlighted source atoms and the vector between them.

2.  **Patterson Map (Vector Space):**
    * **What it is:** The *ideal* Patterson map.
    * **How it's made:** Calculated directly by finding all pairs of atoms in the `Crystal Structure` and plotting a peak for each vector $(x_j - x_i, y_j - y_i)$ with a weight of $f_i \times f_j$.
    * **Interaction:** Hover over a peak to trigger the vector highlighting.

3.  **Diffraction Pattern (|F_hkl|²):**
    * **What it is:** The simulated X-ray diffraction pattern (reciprocal space).
    * **How it's made:** Calculated by taking the Fourier Transform of the `Crystal Structure`. The intensity of each $(h,k)$ spot is $I = |F_{hkl}|^2$.

4.  **FT of Intensities (Patterson):**
    * **What it is:** The Patterson map calculated the "real" way.
    * **How it's made:** Calculated by taking the Fourier Transform of the `Diffraction Pattern` ($|F_{hkl}|^2$ data).
    * **The Point:** This plot should look identical to the `Patterson Map (Vector Space)` plot, demonstrating the mathematical equivalence: $\mathcal{F} ( |F_{hkl}|^2 ) = \text{Vector Map}$.

</section>

---

<section id="algorithms--calculations">

## Algorithms & Calculations 💡

The simulator runs three key calculations in real-time.

### 1. Crystal Structure → Patterson Map

The `Patterson Map (Vector Space)` (Canvas 2) is generated by a direct vector calculation:

1.  An origin peak is created at (0,0) with weight $\sum f_i^2$.
2.  The program iterates through every atom `i` and every atom `j` in the `Atom List`.
3.  For each pair, it calculates the vector:
    * $u = x_j - x_i$
    * $v = y_j - y_i$
4.  It plots a peak at $(u,v)$ with a radius proportional to its weight, $W = f_i \times f_j$.
5.  Peaks are color-coded based on their `vectorType`: 'AA' (green), 'BB' (cyan), or 'AB' (orange).

### 2. Crystal Structure → Diffraction Pattern

The `Diffraction Pattern` (Canvas 3) is generated by calculating the structure factor, $F_{hkl}$, for each $(h,k)$ grid point:

1.  For each $(h,k)$ from `-Range` to `+Range`:
2.  Calculate the real ($A$) and imaginary ($B$) components of the structure factor:
    * $A = \sum f_j \cdot \cos(2\pi(hx_j + ky_j))$
    * $B = \sum f_j \cdot \sin(2\pi(hx_j + ky_j))$
3.  The intensity $I$ is calculated as $I = A^2 + B^2 = |F_{hkl}|^2$.
4.  A spot is drawn at $(h,k)$ with a size and brightness proportional to this intensity.

### 3. Diffraction Pattern → Patterson Map

The `FT of Intensities` (Canvas 4) is generated by taking the Fourier Transform of the diffraction intensities $I_{hk}$ (which are equal to $|F_{hkl}|^2$):

1.  It sums all the diffraction intensities, weighted by a cosine term:
    * $P(u,v) = \sum_h \sum_k I_{hk} \cdot \cos(2\pi(hu + kv))$
2.  This calculation is performed for every pixel $(u,v)$ in the canvas, resulting in an image that is mathematically equivalent to the ideal vector map.

### Interactive Vector Highlighting

This is the key interactive feature of the simulator.

1.  When you move your mouse over the `Patterson Map` (Canvas 2), the program finds the nearest vector peak.
2.  It retrieves the "from" and "to" atom indices stored with that peak.
3.  It then re-draws the `Crystal Structure` (Canvas 1), coloring the "from" atom pink and the "to" atom purple, and draws an arrow between them.
4.  The `Vector Details` box is simultaneously updated with the coordinates of these two atoms and the resulting vector.

</section>

---

<section id="about-this-tool">

## About This Tool

This 2D crystal structure, diffraction pattern, and Patterson function simulator was developed by NitaD, Univ. Paris-Saclay (ver 11 oct 2025).
This documentation was created by an AI assistant on 20 oct 2025.

</section>
