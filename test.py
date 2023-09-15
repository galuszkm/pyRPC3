from SignalReader import RPC3

files = ['SignalExample.rsp']

sig = RPC3(files[0], debug=False)
sig.Channels[-1].plot()
