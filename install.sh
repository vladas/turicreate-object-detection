#!/bin/bash

pip list

ls -la /usr/local/lib/python2.7/dist-packages/turicreate/toolkits/


pip uninstall -y mxnet && pip install mxnet-cu90==1.1.0