from kmcpy.event_generator import generate_events

supercell_shape = (8,8,8)
lce_fname = './inputs/local_cluster_expansion.json'
lce_site_fname = './inputs/local_cluster_expansion.json'

prim_fname = './inputs/prim.json'
event_fname = './events.json'
generate_events(prim_fname,supercell_shape,event_fname)
