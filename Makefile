default: image

image:
	docker build \
		--file docker/Dockerfile \
		--build-arg BASE_IMAGE=scailfin/madgraph5-amc-nlo:mg5_amc3.5.0 \
		--tag madjax-hep/madjax:local \
		.
