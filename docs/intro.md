# What is the punchbowl?

## What is PUNCH?
PUNCH is a NASA Small Explorer (SMEX) mission to better understand how the mass and energy of 
the Sun’s corona become the solar wind that fills the solar system. 
Four suitcase-sized satellites will work together to produce images of the entire inner solar system around the clock.
You can learn more at the [PUNCH website](https://punch.space.swri.edu/). 

## So, where does `punchbowl` fit in? 
`punchbowl` is the data reduction pipeline code for the PUNCH mission. The pipeline, as shown in 
{numref}`Figure {number}<pipeline-schematic>`, consists of several segments of processing. 

```{figure} FO4.png
---
height: 500px
name: pipeline-schematic
---
This is a schematic of the pipeline. 
```

These segments are the following:
- Raw to *Level 0*: converts raw satellite data to FITS images
- Level 0 to *Level 1*: basic image calibration
- Level 1 to *Level 2*: polarization resolution and quality marking
- Level 2 to *Level 3*: background subtraction 

We identify these segments by their finishing level, i.e. the Level 1 products come from the Level 0 to Level 1 segment 
which can be called just the Level 1 segment for short. The processing description and code you'll find here is 
organized in this manner. 