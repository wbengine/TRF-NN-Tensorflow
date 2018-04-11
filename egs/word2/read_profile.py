import pstats
p = pstats.Stats('restats')
p.sort_stats('cumtime').print_stats(20)
