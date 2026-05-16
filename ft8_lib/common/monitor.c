#include "monitor.h"
#include <common/common.h>

#include <stdlib.h>


void monitor_reset(monitor_t* me)
{
    me->wf.num_blocks = 0;
    me->max_mag = -120.0f;
}
