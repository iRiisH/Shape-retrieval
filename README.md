# Shape-retrieval

This project is an attempt in implementing Eitz et al. method ([Sketch-Based Shape Retrieval](http://cybertron.cg.tu-berlin.de/eitz/projects/sbsr/)).

## View selection

In order to select the best view for a given 3D model, we first have to pick a set of views around the object, then select the ones that are likely to be useful. In order to do so, we pick views uniformly at random over the unit circle, then perform Lloyd's relaxation. The importance of this relaxation is illustrated below (left: randomly selected views, right: after relaxation).

![View selection](https://github.com/iRiisH/Shape-retrieval/blob/master/BestDirection/illustr_relaxation.png)
