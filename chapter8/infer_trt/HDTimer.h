#pragma once
#include <time.h>


class HDTimer
{
private:
    struct timespec begin;
    struct timespec end;
    float  elapsed_ms;
public:
    HDTimer(/* args */){};
    ~HDTimer(){};
    void start(){clock_gettime(CLOCK_REALTIME, &begin);}
    void stop(){
            clock_gettime(CLOCK_REALTIME, &end);
            elapsed_ms = 1000.0f*(end.tv_sec - begin.tv_sec) + (end.tv_nsec - begin.tv_nsec)/1000000.0f;
        }

    float elapsed_time_ms(){return elapsed_ms;}
};