## testing madjax

* generate the ME with mg5 using the madjax plugin

```
docker run --rm -it -v `madjax-config`:/code/madgraph/PLUGIN/madjax_me_gen -v $PWD:$PWD -w $PWD lukasheinrich/diffmes /code/madgraph/bin/mg5_aMC --mode=madjax_me_gen dijet.mg5
```

* run a test script

```
python testME.py
```
