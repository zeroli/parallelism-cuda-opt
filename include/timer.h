#ifndef TIMER_H_
#define TIMER_H_

#include <time.h>
#include <sys/time.h>

static double GetTime()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return t.tv_sec * 1.0 + t.tv_nsec / 1000000000.0;
}

#endif
