# go-profile-chart-generator

A tool to generate time series chart from profiles in Go.

Sample result:
![sample](https://user-images.githubusercontent.com/19720977/128287228-99bc3fac-7b21-40a1-81f5-f8a32abf1efd.png)

## Requirements
+ Python
+ Golang

## How to start

Please get profiles before use this tool (Please check: https://blog.golang.org/pprof)

```shell
$ pip install -r requirements.txt
$ python main.py [DIRECTORY_FOR_GO_PROFILES]
```

## Command Option Arguments

+ --out: path of output file
+ --binary: binary file for analysis target
+ --type: type of data, available only if the type of profiles is heap [inuse_space/alloc_space/inuse_objects/alloc_objects]
+ --kind: kind of aggregation
  - flat: value only in the function
  - cum: value in the function and callee functions
+ --num: number of functions to display
+ --title: chart title
+ --ymin: Max limit of y-axis
+ --ymax: Min limit of y-axis
