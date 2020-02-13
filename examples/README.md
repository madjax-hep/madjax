## testing madjax

### Installation

```
pyhton -m venv madjaxenv
source madjaxenv/bin/activate
git clone https://github.com/lukasheinrich/madjax.git
cd madjax
pip install -e .
```


### Generating madjax matrix elements with MadGraph

* generate the ME with mg5 using the madjax plugin and docker

```
docker run --rm -it -v `madjax-config`:/code/madgraph/PLUGIN/madjax_me_gen -v $PWD:$PWD -w $PWD lukasheinrich/diffmes /code/madgraph/bin/mg5_aMC --mode=madjax_me_gen higgs.mg5
```

* .. or using a local install of MadGraph (might require its own python env)

```
tar -xzvf MG5_aMC_v2.7.0.tar.gz 
ln -s $(madjax-config) MG5_aMC_v2_7_0/PLUGIN/
# in a python 2.7 shell
PYTHONPATH=/path/to/MG5_aMC_v2_7_0/PLUGIN /path/to/MG5_aMC_v2_7_0/bin/mg5_aMC --mode=madjax_me_gen higgs.mg5
```


* run a test script
```
python testME.py higgs4l
```
