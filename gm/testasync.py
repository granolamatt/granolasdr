import numpy as np
import granolasdr as gsdr
import asyncio


async def get_await_result(x):
    val = await x
    print("val got",val)
    return val

async def printit():
    for cnt in range(100):
        await asyncio.sleep(0.1)
        print("printit")


async def main():
    # async with asyncio.TaskGroup() as tg:
    #     task1 = tg.create_task(get_await_result(gsdr.SupportsAsync()))
    #     task2 = tg.create_task(printit())

    # print("Returned", task1.result())

    tasks = [asyncio.create_task(get_await_result(gsdr.SupportsAsync())), asyncio.create_task(printit())]
    result = await asyncio.gather(*tasks)
    print(result)

    # val = await get_await_result(gsdr.SupportsAsync())

    # print(val)

asyncio.run(main())
