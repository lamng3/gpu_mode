# Notes

CUDA is async so cannot use Python time module. You can only measure the overhead time it takes to launch a kernel, not the actual time the kernel actually takes to run. This is a **key thing to remember* about profiling CUDA.