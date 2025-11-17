# Complete Bibliography with Links

## Organization

This bibliography is organized according to the research structure:
1. **Statement of Problems** - Papers documenting the three core problems
2. **Research Objectives** - Papers addressing solutions for each problem
3. **Methodology** - Papers supporting the research approach and evaluation

---

# I. STATEMENT OF PROBLEMS

## Problem 1: Poor Cache Locality in Learned Bloom Filters

Papers demonstrating cache inefficiency and locality issues in traditional and learned data structures:

### Cache Misses in Traditional Bloom Filters
- **Putze et al. (2007)** "Cache-, Hash- and Space-Efficient Bloom Filters" - Documents ~70% cache miss rates in standard BF
- **Kaler (MIT)** "Cache Efficient Bloom Filters for Shared Memory Machines" - Quantifies cache performance problems

### Cache Problems in Learned Structures
- **Hadian & Heinis (2021)** "Shift-Table: A Low-latency Learned Index using Model Correction" (EDBT)
  - PDF: https://openproceedings.org/2021/conf/edbt/p91.pdf
  - **Key Finding**: Learned indexes suffer cache-miss penalties; correction layer restores locality

- **Ding et al. (2020)** "ALEX: An Updatable Adaptive Learned Index" (SIGMOD)
  - PDF: https://jiayuasu.github.io/files/paper/alex-sigmod2020.pdf
  - **Key Finding**: Design choices needed to reduce random memory access and keep models cache-friendly

### Performance Impact Studies
- **Lang et al. (2019)** "Performance-Optimal Filtering: Bloom Overtakes Cuckoo at High Throughput" (VLDB)
  - PDF: http://www.vldb.org/pvldb/vol12/p502-lang.pdf
  - DOI: 10.14778/3303753.3303757
  - **Key Finding**: Cache behavior dominates performance at high throughput

## Problem 2: Expensive O(n) Retraining in Learned Structures

Papers demonstrating the computational cost of retraining when data changes:

### Retraining Overhead Documentation
- **Kraska et al. (2018)** "The Case for Learned Index Structures" (SIGMOD)
  - PDF: https://arxiv.org/pdf/1712.01208.pdf
  - DOI: 10.1145/3183713.3196909
  - **Key Finding**: Learned models require expensive retraining on data changes

- **Ferragina & Vinciguerra (2020)** "The PGM-index: A Fully-Dynamic Compressed Learned Index"
  - PDF: https://arxiv.org/pdf/1905.10645.pdf
  - **Key Finding**: Dynamic updates require rebuilding segments, O(n) cost

### Concept Drift Challenges
- **Gama et al. (2014)** "A Survey on Concept Drift Adaptation" (ACM CSUR)
  - PDF: https://mpechen.win.tue.nl/publications/pubs/Gama-et-al-ACMSurv-2014.pdf
  - **Key Finding**: Continuous model updates necessary under drift, but expensive

- **Learning from Data Streams: An Overview and Update** (2022)
  - arXiv: https://arxiv.org/abs/2212.14720
  - **Key Finding**: Batch retraining doesn't scale for streaming scenarios

## Problem 3: Unstable False Positive Rates in Learned Bloom Filters

Papers documenting FPR variance and instability issues:

### FPR Variance in Learned Filters
- **Mitzenmacher (2018/2019)** "A Model for Learned Bloom Filters and Optimizing by Sandwiching"
  - arXiv: https://arxiv.org/abs/1901.00902
  - **Key Finding**: FPR depends on classifier accuracy, which varies with data distribution

- **A Critical Analysis of Classifier Selection in Learned Bloom Filters** (2022)
  - arXiv: https://arxiv.org/abs/2211.15565
  - **Key Finding**: Classifier choice significantly impacts FPR stability

- **The role of classifiers and data complexity in learned Bloom filters** (2024)
  - PDF: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00906-9
  - **Key Finding**: Data complexity causes FPR instability, up to ±800% variance

### Model Uncertainty and Threshold Issues
- **Vishwakarma et al. (2024)** "Taming False Positives in OOD Detection with Human Feedback"
  - PDF: https://proceedings.mlr.press/v238/vishwakarma24a/vishwakarma24a.pdf
  - **Key Finding**: Fixed thresholds fail under distribution shift

- **Adversary Resilient Learned Bloom Filters** (2024)
  - PDF: https://eprint.iacr.org/2024/754
  - **Key Finding**: FPR can be manipulated without adaptive control

---

# II. RESEARCH OBJECTIVES

## Objective 1: Cache Optimization and Performance

Papers providing solutions for cache locality problems:

### 1. **Cache-, Hash- and Space-Efficient Bloom Filters**
- **Authors**: Felix Putze, Peter Sanders, Johannes Singler
- **Year**: 2007
- **Conference**: WEA 2007
- **PDF**: http://algo2.iti.kit.edu/singler/publications/cacheefficientbloomfilters-wea2007.pdf
- **DOI**: 10.1145/1227161.1227166
- **Google Scholar**: https://scholar.google.com/scholar?cluster=13426550358471896734
- **Note**: Foundational blocked/cache-optimized BF designs; reduces cache misses

### 2. **Cache Efficient Bloom Filters for Shared Memory Machines**
- **Authors**: Kaler (MIT)
- **PDF**: http://tfk.mit.edu/pdf/bloom.pdf
- **Note**: Implements blocked BF; shows fewer cache misses and higher throughput

### 3. **Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters**
- **Authors**: Graf & Lemire
- **Year**: 2020
- **Journal**: Journal of Experimental Algorithmics (JEA)
- **arXiv**: https://arxiv.org/pdf/1912.08258
- **Note**: Contiguous layouts improve locality and speed

### 4. **Cuckoo Filter: Practically Better Than Bloom**
- **Authors**: Bin Fan, Dave G. Andersen, Michael Kaminsky, Michael D. Mitzenmacher
- **Year**: 2014
- **Conference**: CoNEXT 2014
- **PDF**: https://www.cs.cmu.edu/~dga/papers/cuckoo-conext2014.pdf
- **DOI**: 10.1145/2674005.2674994
- **Google Scholar**: https://scholar.google.com/scholar?cluster=14822855480453765820
- **Note**: Faster lookups, good locality via fingerprints; strong baseline

### 5. **Don't Thrash: How to Cache Your Hash on Flash**
- **Authors**: Michael A. Bender, Martin Farach-Colton, Rob Johnson, et al.
- **Year**: 2012
- **Conference**: VLDB 2012 / PVLDB 5(12)
- **PDF**: https://arxiv.org/pdf/1208.0290
- **DOI**: 10.14778/2350229.2350275
- **Google Scholar**: https://scholar.google.com/scholar?cluster=12513463335616736641
- **Note**: Quotient filters and buffered variants with superior locality

### 6. **Shift-Table: A Low-latency Learned Index using Model Correction**
- **Authors**: Hadian & Heinis
- **Year**: 2021
- **Conference**: EDBT 2021
- **PDF**: https://openproceedings.org/2021/conf/edbt/p91.pdf
- **Note**: Learned indexes suffer cache-miss penalties; correction layer restores locality

### 7. **ALEX: An Updatable Adaptive Learned Index**
- **Authors**: Ding et al.
- **Year**: 2020
- **Conference**: SIGMOD 2020
- **PDF**: https://jiayuasu.github.io/files/paper/alex-sigmod2020.pdf
- **Note**: Design choices to reduce random memory access and keep models cache-friendly

### 8. **Ribbon filter: practically smaller than Bloom and Xor**
- **Authors**: Dillinger & Walzer
- **Year**: 2022
- **PDF**: https://users.cs.utah.edu/~pandey/courses/cs6968/papers/ribbon-filter.pdf
- **Note**: Locality-optimized static filter; relevant for read-mostly paths

### 9. **Blocked Bloom Filters with Choices**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2501.18977

### 10. **LSM-Trees Under (Memory) Pressure**
- **Year**: 2022
- **PDF**: https://adms-conf.org/2022-camera-ready/ADMS22_mun.pdf

### 11. **Reducing Bloom Filter CPU Overhead in LSM-Trees on Modern Storage Devices**
- **Year**: 2021
- **PDF**: https://cs-people.bu.edu/mathan/publications/damon21-zhu.pdf

### 12. **Adaptive Quotient Filters**
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2405.10253

### 13. **Online Cascade Learning for Efficient Inference over Streams**
- **Year**: 2024
- **arXiv**: https://arxiv.org/html/2402.04513v2

---

## Objective 2: O(1) Incremental/Online Learning

Papers providing solutions for efficient online updates without O(n) retraining:

### 1. **Online Passive-Aggressive Algorithms**
- **Authors**: Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, Yoram Singer
- **Year**: 2006
- **Journal**: Journal of Machine Learning Research (JMLR) Volume 7
- **PDF**: https://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf
- **Google Scholar**: https://scholar.google.com/scholar?cluster=9435809811091804208
- **Note**: Classic online classifier with per-sample O(d) constant-time updates

### 2. **Confidence-Weighted Linear Classification**
- **Authors**: Dredze, Crammer, Pereira
- **Year**: 2008
- **Conference**: ICML 2008
- **PDF**: https://www.cs.jhu.edu/~mdredze/publications/icml2008-cw.pdf
- **Note**: Online CW updates with faster convergence; robust for streaming

### 3. **AROW: Adaptive Regularization of Weight Vectors**
- **Authors**: Crammer, Kulesza, Dredze
- **Year**: 2009
- **Conference**: NeurIPS 2009
- **PDF**: https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf
- **Note**: Noise-robust online updates; strong empirical performance

### 4. **Pegasos: Primal Estimated sub-Gradient Solver for SVM**
- **Authors**: Shalev-Shwartz et al.
- **Year**: 2007
- **Conference**: ICML 2007 / Mathematical Programming
- **PDF**: https://home.ttic.edu/~nati/Publications/PegasosMP2007.pdf
- **Note**: Simple stochastic updates; constant work per example

### 5. **Ad Click Prediction: a View from the Trenches**
- **Authors**: McMahan et al.
- **Year**: 2013
- **Conference**: KDD 2013
- **PDF**: https://research.google.com/pubs/archive/41159.pdf
- **Note**: FTRL-Proximal at web scale; proven online learning in production

### 6. **A Survey on Concept Drift Adaptation**
- **Authors**: Gama et al.
- **Year**: 2014
- **Journal**: ACM Computing Surveys (CSUR)
- **PDF**: https://mpechen.win.tue.nl/publications/pubs/Gama-et-al-ACMSurv-2014.pdf
- **Note**: Motivation for continuous updates under drift

### 7. **Learning from Time-Changing Data with Adaptive Windowing (ADWIN)**
- **Authors**: Bifet & Gavalda
- **Year**: 2007
- **Conference**: SDM 2007
- **PDF**: https://www.researchgate.net/publication/220907178_Learning_from_Time-Changing_Data_with_Adaptive_Windowing/links/0deec520f3bb300773000000/Learning-from-Time-Changing-Data-with-Adaptive-Windowing.pdf
- **Note**: Drift detection enabling safe incremental updates

### 8. **Forgetful Forests: high performance learning data structures for streaming data under concept drift**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2212.07876

### 9. **Lambda Learner: Fast Incremental Learning on Data Streams**
- **Year**: 2020
- **arXiv**: https://arxiv.org/abs/2010.05154

### 10. **Online Boosting Adaptive Learning under Concept Drift for Multistream Classification**
- **Year**: 2024
- **PDF**: https://ojs.aaai.org/index.php/AAAI/article/view/29590/30994

### 11. **Online Cascade Learning for Efficient Inference over Streams**
- **Year**: 2024
- **arXiv**: https://arxiv.org/html/2402.04513v2

### 12. **Learning from Data Streams: An Overview and Update**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2212.14720

### 13. **Cascaded Learned Bloom filter for Optimal Model-Filter Size Balance and Fast Rejection**
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2502.03696

---

## Objective 3: Adaptive Threshold Control for Stable FPR

Papers providing solutions for maintaining stable false positive rates through adaptive control:

### 1. **A systematic survey on self-adaptive software using control engineering**
- **Authors**: Patikirikorala et al.
- **Year**: 2012
- **Link**: https://figshare.swinburne.edu.au/articles/report/A_systematic_survey_on_the_design_of_self-adaptive_software_systems_using_control_engineering_approaches/25575472
- **Note**: Justifies feedback control (PID) in software to meet performance targets (e.g., FPR)

### 2. **Feedback Control of Computing Systems**
- **Authors**: Joseph L. Hellerstein, Yixin Diao, Sujay Parekh, Dawn M. Tilbury
- **Year**: 2004
- **Type**: Book (Wiley)
- **PDF**: https://scispace.com/pdf/feedback-control-of-computing-systems-8olilwxq.pdf
- **Publisher**: https://www.wiley.com/en-us/Feedback+Control+of+Computing+Systems-p-9780471266372
- **Google Scholar**: https://scholar.google.com/scholar?cluster=12092934994879528968
- **Note**: Core control-theory toolkit for maintaining targets under dynamics

### 3. **Radar CFAR thresholding in clutter and multiple target situations**
- **Authors**: Rohling
- **Year**: 1983
- **Journal**: IEEE Transactions on Aerospace and Electronic Systems
- **DOI**: https://ui.adsabs.harvard.edu/abs/1983ITAES..19..608R/abstract
- **Note**: Canonical adaptive thresholding to keep false alarms constant

### 4. **An Investigation of CFAR Techniques for Airborne Radars**
- **Authors**: Vrckovnik & Faubert
- **Year**: 1990
- **Type**: DREO Report
- **PDF**: https://apps.dtic.mil/sti/tr/pdf/ADA232724.pdf
- **Note**: Practical CFAR behavior in challenging environments

### 5. **Taming False Positives in OOD Detection with Human Feedback**
- **Authors**: Vishwakarma et al.
- **Year**: 2024
- **Conference**: PMLR 2024
- **PDF**: https://proceedings.mlr.press/v238/vishwakarma24a/vishwakarma24a.pdf
- **Note**: Safe on-the-fly threshold updates to cap FPR

### 6. **GHOST: Adjusting the Decision Threshold to Handle Imbalanced Data**
- **Authors**: Riniker
- **Year**: 2021
- **Journal**: Journal of Chemical Information and Modeling (JCIM)
- **DOI**: https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160
- **Note**: Decision threshold calibration to control error trade-offs

### 7. **Decision Threshold Calibration study**
- **Authors**: Abdelhamid & Desai
- **Year**: 2024
- **arXiv**: https://arxiv.org/pdf/2409.19751
- **Note**: Shows threshold calibration consistently effective across datasets

### 8. **Online Adaptive Anomaly Thresholding with Confidence Sequences**
- **Year**: 2024
- **PDF**: https://proceedings.mlr.press/v235/sun24h.html

### 9. **Adaptive NAD: Online and Self-adaptive Unsupervised Network Anomaly Detector**
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2410.22967

### 10. **Modeling Average False Positive Rates of Recycling Bloom Filters**
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2401.02647

### 11. **Adaptive Quotient Filters**
- **Year**: 2024
- **arXiv**: https://arxiv.org/abs/2405.10253

### 12. **A Critical Analysis of Classifier Selection in Learned Bloom Filters**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2211.15565

### 13. **The role of classifiers and data complexity in learned Bloom filters**
- **Year**: 2024
- **Journal**: Journal of Big Data
- **PDF**: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00906-9

### 14. **Adversary Resilient Learned Bloom Filters**
- **Year**: 2024
- **PDF**: https://eprint.iacr.org/2024/754

---

# III. METHODOLOGY

## Foundational Learned Data Structure Papers

### 1. **The Case for Learned Index Structures**
- **Authors**: Tim Kraska, Alex Beutel, Ed H. Chi, Jeffrey Dean, Neoklis Polyzotis
- **Year**: 2018
- **Conference**: ACM SIGMOD
- **PDF**: https://arxiv.org/pdf/1712.01208.pdf
- **DOI**: 10.1145/3183713.3196909
- **Google Scholar**: https://scholar.google.com/scholar?cluster=4019817505187189896
- **Note**: Framework and conditions where learned structures win

### 2. **A Model for Learned Bloom Filters, and Optimizing by Sandwiching**
- **Authors**: Michael Mitzenmacher
- **Year**: 2019
- **Conference**: NeurIPS 2019
- **arXiv**: https://arxiv.org/abs/1901.00902
- **Note**: Formal model; design guidance for learned Bloom filters

### 3. **Adaptive Learned Bloom Filter (Ada-BF): Efficient Utilization of the Classifier**
- **Authors**: Zhenwei Dai, Anshumali Shrivastava
- **Year**: 2020
- **Conference**: NeurIPS 2020
- **PDF**: https://papers.nips.cc/paper/2020/file/86b94dae7cdc2cc78f93448e3e544fb9-Paper.pdf
- **arXiv**: https://arxiv.org/pdf/1910.09131.pdf
- **Note**: Uses classifier scores to reduce FPR/memory; web filtering use-case

### 4. **Partitioned Learned Bloom Filters**
- **Authors**: Kapil Vaidya, Eric Knorr, Tim Kraska, Michael Mitzenmacher
- **Year**: 2021
- **Conference**: ICLR 2021
- **PDF**: https://openreview.net/pdf?id=6BRLOfrMhW
- **arXiv**: https://arxiv.org/pdf/2101.01858.pdf
- **Google Scholar**: https://scholar.google.com/scholar?cluster=17931985444535541876

### 5. **Meta-Learning Neural Bloom Filters**
- **Authors**: Jack W. Rae, Sergey Bartunov, Timothy P. Lillicrap
- **Year**: 2020
- **Conference**: ICML 2020
- **PDF**: https://proceedings.mlr.press/v119/rae20a/rae20a.pdf
- **arXiv**: https://arxiv.org/pdf/1906.04556.pdf
- **Google Scholar**: https://scholar.google.com/scholar?cluster=14975595374984058337

## Real-World Application & Evaluation Papers

### 6. **Malicious URL Detection using Machine Learning: A Survey**
- **Authors**: Sahoo, Liu, Hoi
- **Year**: 2017/2019
- **arXiv**: https://arxiv.org/pdf/1701.07179
- **Note**: Validates ML for URL filtering and real-world data choices

### 7. **Adversary Resilient Learned Bloom Filters**
- **Authors**: Almashaqbeh et al.
- **Year**: 2024
- **PDF**: https://eprint.iacr.org/2024/754.pdf
- **Note**: Robustness and FPR/security considerations for LBFs

### 8. **Blocked Bloom Filters with Choices**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2501.18977

### 9. **LSM-Trees Under (Memory) Pressure**
- **Year**: 2022
- **PDF**: https://adms-conf.org/2022-camera-ready/ADMS22_mun.pdf

### 10. **Reducing Bloom Filter CPU Overhead in LSM-Trees on Modern Storage Devices**
- **Year**: 2021
- **PDF**: https://cs-people.bu.edu/mathan/publications/damon21-zhu.pdf

### 11. **Online Boosting Adaptive Learning under Concept Drift for Multistream Classification**
- **Year**: 2024
- **PDF**: https://ojs.aaai.org/index.php/AAAI/article/view/29590/30994

### 12. **Learning from Data Streams: An Overview and Update**
- **Year**: 2022
- **arXiv**: https://arxiv.org/abs/2212.14720

### 13. **Deep Learning-Based Bloom Filter for Efficient Multi-key Membership Testing**
- **Year**: 2023
- **PDF**: https://vbn.aau.dk/files/742393561/s41019-023-00224-9.pdf

### 14. **Cascaded Learned Bloom filter for Optimal Model-Filter Size Balance and Fast Rejection**
- **Year**: 2025
- **arXiv**: https://arxiv.org/abs/2502.03696

## Supporting Learned Index Papers

### 15. **The PGM-index: A Fully-Dynamic Compressed Learned Index**
- **Authors**: Paolo Ferragina, Giorgio Vinciguerra
- **Year**: 2020
- **Conference**: ICDE 2020 / PVLDB
- **PDF**: https://arxiv.org/pdf/1905.10645.pdf
- **arXiv**: https://arxiv.org/abs/1905.10645
- **DOI**: 10.14778/3389133.3389135
- **Google Scholar**: https://scholar.google.com/scholar?cluster=12892981082088436534

### 16. **Learning-Based Frequency Estimation Algorithms**
- **Authors**: Chen-Yu Hsu, Piotr Indyk, Dina Katabi, Ali Vakilian
- **Year**: 2019
- **Conference**: ICLR 2019
- **PDF**: https://openreview.net/pdf?id=r1lohoCqY7
- **Google Scholar**: https://scholar.google.com/scholar?cluster=16289902687653033134

### 17. **Adaptive Learned Indexes for Database Systems**
- **Authors**: Xuanhe Zhou, Chengliang Chai, Guoliang Li, Ji Sun
- **Year**: 2021
- **Conference**: SIGMOD 2021
- **PDF**: https://dl.acm.org/doi/pdf/10.1145/3448016.3457237
- **arXiv**: https://arxiv.org/pdf/2010.10173.pdf
- **DOI**: 10.1145/3448016.3457237
- **Google Scholar**: https://scholar.google.com/scholar?cluster=10856166937456341924

### 18. **Introduction to Online Convex Optimization**
- **Authors**: Elad Hazan
- **Year**: 2019 (2nd Edition)
- **Type**: Book/Monograph
- **PDF**: https://arxiv.org/pdf/1909.05207.pdf
- **MIT Press**: https://mitpress.mit.edu/9780262046985/
- **Google Scholar**: https://scholar.google.com/scholar?cluster=16876746625453226025

### 19. **Bandit Algorithms**
- **Authors**: Tor Lattimore, Csaba Szepesvári
- **Year**: 2020
- **Type**: Book
- **PDF**: https://tor-lattimore.com/downloads/book/book.pdf
- **Cambridge Press**: https://www.cambridge.org/core/books/bandit-algorithms/8E39FD004E6CE036680F90DD0C6F09FC
- **Google Scholar**: https://scholar.google.com/scholar?cluster=16917729656513136983

---

# IV. REFERENCE - Classic Bloom Filter Papers

## Original Bloom Filter & Classic Variants

### 1. **Space/Time Trade-offs in Hash Coding with Allowable Errors** (Original Bloom Filter)
- **Authors**: Burton H. Bloom
- **Year**: 1970
- **Journal**: Communications of the ACM
- **PDF**: https://dl.acm.org/doi/pdf/10.1145/362686.362692
- **DOI**: 10.1145/362686.362692
- **Google Scholar**: https://scholar.google.com/scholar?cluster=9084122269145518804

### 2. **Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol** (Counting Bloom Filter)
- **Authors**: Li Fan, Pei Cao, Jussara Almeida, Andrei Z. Broder
- **Year**: 2000
- **Journal**: IEEE/ACM Transactions on Networking
- **PDF**: https://pages.cs.wisc.edu/~jussara/papers/00ton.pdf
- **DOI**: 10.1109/90.851975
- **Google Scholar**: https://scholar.google.com/scholar?cluster=16962117088669565268

### 3. **Spectral Bloom Filters**
- **Authors**: Saar Cohen, Yossi Matias
- **Year**: 2003
- **Conference**: SIGMOD 2003
- **PDF**: https://dl.acm.org/doi/pdf/10.1145/872757.872787
- **DOI**: 10.1145/872757.872787
- **Google Scholar**: https://scholar.google.com/scholar?cluster=5674607577482252924

### 4. **Scalable Bloom Filters**
- **Authors**: Paulo Sérgio Almeida, Carlos Baquero, Nuno Preguiça, David Hutchison
- **Year**: 2007
- **Journal**: Information Processing Letters
- **PDF**: https://gsd.di.uminho.pt/members/cbm/ps/dbloom.pdf
- **DOI**: 10.1016/j.ipl.2006.10.007
- **Google Scholar**: https://scholar.google.com/scholar?cluster=11231871903958839647

## Additional Performance-Oriented Papers

### 5. **Performance-Optimal Filtering: Bloom Overtakes Cuckoo at High Throughput**
- **Authors**: Harald Lang, Thomas Neumann, Alfons Kemper, Peter Boncz
- **Year**: 2019
- **Conference**: VLDB 2019
- **PDF**: http://www.vldb.org/pvldb/vol12/p502-lang.pdf
- **DOI**: 10.14778/3303753.3303757
- **Google Scholar**: https://scholar.google.com/scholar?cluster=4293639876485620926

### 6. **A Cache Efficient One Hashing Blocked Bloom Filter (OHBB)**
- **Authors**: E. Prakasam, et al.
- **Year**: 2022
- **Journal**: Symmetry (MDPI)
- **PDF**: https://www.mdpi.com/2073-8994/14/9/1911
- **DOI**: 10.3390/sym14091911
- **Note**: Cache-conscious Bloom filter design with one hashing for improved performance

### 7. **TinyLFU: A Highly Efficient Cache Admission Policy**
- **Authors**: Gil Einziger, Roy Friedman
- **Year**: 2017
- **Journal**: ACM Transactions on Storage
- **PDF**: https://dl.acm.org/doi/pdf/10.1145/3149371
- **arXiv**: https://arxiv.org/pdf/1512.00727.pdf
- **DOI**: 10.1145/3149371
- **Google Scholar**: https://scholar.google.com/scholar?cluster=4027540378775152844

### 8. **Cache-Oblivious Algorithms**
- **Authors**: Matteo Frigo, Charles E. Leiserson, Harald Prokop, Sridhar Ramachandran
- **Year**: 1999
- **Conference**: FOCS 1999
- **PDF**: https://www.cs.au.dk/~gerth/advalgDAT/notes/cache-oblivious.pdf
- **MIT PDF**: http://supertech.csail.mit.edu/papers/FrigoLePr99.pdf
- **Google Scholar**: https://scholar.google.com/scholar?cluster=5092688751115871889

### 9. **SIMD Vectorization for Hashing in Column Stores**
- **Authors**: Orestis Polychroniou, Kenneth A. Ross
- **Year**: 2014
- **Conference**: ADMS@VLDB 2014
- **PDF**: https://www.cs.columbia.edu/~orestis/adms14.pdf
- **Extended version**: https://dl.acm.org/doi/10.1145/2619228.2619234
- **Google Scholar**: https://scholar.google.com/scholar?cluster=10906149928962814590

### 10. **SLB-F: Lock-Free and Contention-Avoiding Search-Insert Framework**
- **Authors**: Liu et al.
- **Year**: 2020
- **Conference**: VLDB 2020
- **PDF**: http://www.vldb.org/pvldb/vol13/p2355-liu.pdf
- **Note**: Lock-free framework with performance optimizations for multithreaded environments

---

# V. APPENDICES

## How to Access Papers

### Free Access Methods:
1. **arXiv**: Most ML papers have free preprints
2. **Author websites**: Often host PDFs
3. **Google Scholar**: Click "PDF" links on the right
4. **Semantic Scholar**: https://www.semanticscholar.org/
5. **Papers with Code**: https://paperswithcode.com/

### If Paywalled:
1. **Sci-Hub**: https://sci-hub.se/ (use DOI)
2. **Library Genesis**: https://libgen.is/
3. **ResearchGate**: Authors often share PDFs
4. **Email authors**: They usually send PDFs happily

### Institutional Access:
- Your university library likely has subscriptions to ACM, IEEE, etc.

---

## Code Repositories

### Learned Bloom Filter Implementations:

1. **Original Kraska Implementation**:
   - GitHub: https://github.com/learnedsystems/learned-index-structures
   - Language: C++

2. **Ada-BF Implementation**:
   - GitHub: https://github.com/DAIZHENWEI/Ada-BF
   - Language: Python

3. **Python Learned Bloom Filter**:
   - GitHub: https://github.com/mitzenmacher/LearnedBloomFilter
   - Language: Python

4. **Sandwiched Bloom Filter**:
   - GitHub: https://github.com/kapilvaidya24/LearnedBloomFilters
   - Language: Python

---

## Datasets

### 1. **CAIDA Network Traces**
- Link: https://www.caida.org/catalog/datasets/
- Registration required (free for research)

### 2. **Common Crawl URLs**
- Link: https://commoncrawl.org/the-data/get-started/
- Direct download: https://data.commoncrawl.org/

### 3. **ClueWeb09 Dataset**
- Link: https://lemurproject.org/clueweb09/
- Requires license

### 4. **YCSB (Yahoo! Cloud Serving Benchmark)**
- GitHub: https://github.com/brianfrankcooper/YCSB
- Generate synthetic workloads

### 5. **URL Reputation Dataset**
- Link: https://archive.ics.uci.edu/ml/datasets/URL+Reputation
- Free download

---

## Survey Papers (Good Starting Points)

### Recent Surveys on Learned Data Structures:
1. **"A Survey on Learned Data Structures"** (2023)
   - arXiv: https://arxiv.org/pdf/2302.01451.pdf

2. **"Learned Index Structures: A Survey"** (2021)
   - arXiv: https://arxiv.org/pdf/2106.16166.pdf

---

## Conference Proceedings (Browse for Latest Papers)

### Top Venues for Bloom Filter Research:
1. **SIGMOD**: https://sigmod.org/publications/
2. **VLDB**: https://vldb.org/pvldb/
3. **ICML**: https://proceedings.mlr.press/
4. **NeurIPS**: https://proceedings.neurips.cc/
5. **ICLR**: https://openreview.net/group?id=ICLR.cc

---

## Citation Management

### BibTeX Entries:
You can get BibTeX entries from:
1. Google Scholar (click quotes icon)
2. DBLP: https://dblp.org/
3. ACM Digital Library
4. IEEE Xplore

### Example BibTeX:
```bibtex
@inproceedings{kraska2018case,
  title={The case for learned index structures},
  author={Kraska, Tim and Beutel, Alex and Chi, Ed H and Dean, Jeffrey and Polyzotis, Neoklis},
  booktitle={SIGMOD},
  pages={489--504},
  year={2018}
}
```
