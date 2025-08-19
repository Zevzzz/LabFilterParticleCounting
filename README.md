# Lab Filter Particle Counter
**Code to count particle number on lab filters** 

## Team
- **Donson Xie**
- Andrew Oldag 

## Significant Features
- Extracts number of particles from an image of a lab filter
- Preview of detected particles will be shown
- Outputs are saved to an outputs folder as both coordinates and snapshots 

## Instructions
- Upload sample image as `sampleImg.jpg` within the `src/samples` folder
- Run `main.py`, wait for processing
- A preview image will be shown with detected particles 
- Outputs will be saved to `outputs` under a folder named `month-day-year_hour-min_sec`, as:
  - `Output.txt`, which stores the particle count, and for each particle the label, width, and height
  - An `images` folder, with `.jpg` snapshots of detected particles, with 100px margins and boxed in red

