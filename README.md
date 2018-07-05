RankLib
=============

## Overview

RankLib is a library of learning to rank algorithms. Currently eight popular algorithms have been implemented:

* MART (Multiple Additive Regression Trees, a.k.a. Gradient boosted regression tree) [6]
* RankNet [1]
* RankBoost [2]
* AdaRank [3]
* Coordinate Ascent [4]
* LambdaMART [5]
* ListNet [7]
* Random Forests [8]

It also implements many retrieval metrics as well as provides many ways to carry out evaluation.

This project forked from [The Lemur Project](https://sourceforge.net/p/lemur/wiki/RankLib/).

## Version

[Versions in Maven Repository](http://central.maven.org/maven2/org/codelibs/ranklib/)

## License

RankLib is available under BSD license.

## References

1. C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton and G. Hullender. Learning to rank using gradient descent. In Proc. of ICML, pages 89-96, 2005.
2. Y. Freund, R. Iyer, R. Schapire, and Y. Singer. An efficient boosting algorithm for combining preferences. The Journal of Machine Learning Research, 4: 933-969, 2003.
3. J. Xu and H. Li. AdaRank: a boosting algorithm for information retrieval. In Proc. of SIGIR, pages 391-398, 2007.
4. D. Metzler and W.B. Croft. Linear feature-based models for information retrieval. Information Retrieval, 10(3): 257-274, 2007.
5. Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures. Journal of Information Retrieval, 2007.
6. J.H. Friedman. Greedy function approximation: A gradient boosting machine. Technical Report, IMS Reitz Lecture, Stanford, 1999; see also Annals of Statistics, 2001.
7. Z. Cao, T. Qin, T.Y. Liu, M. Tsai and H. Li. Learning to Rank: From Pairwise Approach to Listwise Approach. ICML 2007. 
8. L. Breiman. Random Forests. Machine Learning 45 (1): 5–32, 2001.

