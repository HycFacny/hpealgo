all:
	cd lib/utils/nms; python setup_linux.py build_ext --inplace; rm -rf build; cd ../../../

clean:
	cd lib/utils/nms; rm *.so; cd ../../../
