#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "gm/Thread.h"

namespace py = pybind11;

PYBIND11_MODULE(granolasdr, m) {
    m.doc() = "granola sdr bindings";

    py::class_<gm::Runnable> gmrunnable(m, "Runnable");
    gmrunnable
    .def("setRunning", &gm::Runnable::setRunning)
    .def("isRunning", &gm::Runnable::isRunning);

    py::class_<gm::Thread> gmthread(m, "Thread", gmrunnable);
    gmthread
    .def("start", &gm::Thread::start)
    .def("join", &gm::Thread::join);

    /**
    Right out of pybind11 tests
     */

    struct SupportsAsync {};
    py::class_<SupportsAsync>(m, "SupportsAsync")
    .def(py::init<>())
    .def("__await__", [](const SupportsAsync &self) -> py::object {
        
        static_cast<void>(self);
        py::object loop = py::module_::import("asyncio.events").attr("get_event_loop")();
        py::object f = loop.attr("create_future")();

        gm::Thread *tester = new gm::Thread([=] {
            //printf("Tester here\n");
            // When tester falls out of scope the destructor will be called and stop the while loop
            for (int cnt = 0; cnt < 1; cnt++) {
                printf("Sleeping\n");
                usleep(1000000);
            }
            printf("fell out of scope so told to stop\n");
            py::gil_scoped_acquire acquire;
            printf("Got the gill\n");
            bool loop_is_running = py::cast<bool>(loop.attr("is_running")());
            f.attr("set_result")(5);
            printf("Result is set %u\n", loop_is_running);
            });
            tester->start();
            printf("Sending a wait\n");
            // f.attr("set_result")(5);
            return f.attr("__await__")();
        });

}

