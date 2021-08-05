# go-profile-chart-generator

A tool to generate time series chart from profiles in Go.

Sample result:
![output](https://user-images.githubusercontent.com/19720977/128286993-cbb3aeb8-d836-48fe-8b49-b97534aaeb6f.png)


## How to start

Need to collect profiles before use this tool (Please check: https://blog.golang.org/pprof)

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
