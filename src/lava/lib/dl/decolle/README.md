# DECOLLE

Deep Continuous Local Learning (DECOLLE) is a surrogate gradient (SG) 
algorithm leveraging per-layer local errors for online learning of 
deep SNNs, for which details can be found in reference
[Kaiser et al., 2020](https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full). 

Each layer of the SNN is associated to a fixed, random projection layer
to compute the real-valued errors.

This implementation is based on the authors', which can be found on
their [GitHub repository](https://github.com/nmi-lab/decolle-public/tree/master). 

