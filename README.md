# napari-topograph
Quality checking for whole-slide images.

The primary use case for this is checking the staining quality of multiplexed IHC/IF images. Because there is a discrepancy in staining quality along a slide, this plugin allows investigating the histograms of each channel, and 2d correlation histograms between channels.

Brightfield scn images consist of a background pyramid with one or more pyramids.

MxIF scn images consist of one or more pyramids.

NOTE: not an actual napari plugin (yet)

Run `bin/run.py`.
