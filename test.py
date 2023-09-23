from pyRPC3 import RPC3

files = ['SignalExample.rsp']

sig = RPC3(files[0], debug=False)
sig.Channels[-1].plot()
sig.Channels[-1].rainflow_counting()
sig.Channels[-1].calculate_damage(5, gate=5, plot_graphs=True)

# Save rsp file
sig.save('OutputSignal.rsp', sig.dt, sig.Channels)
