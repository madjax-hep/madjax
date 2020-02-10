import pkg_resources


def cli():
    print(pkg_resources.resource_filename('madjax', 'mg5/madjax_me_gen'))
