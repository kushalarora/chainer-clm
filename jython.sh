#!/bin/bash
java -Xmx6g -jar dependencies/jython-standalone-2.7.0.jar -Dpython.path dependencies/stanford-parser.jar $*
