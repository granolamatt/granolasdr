#ifndef _GM_THREAD_H_
#define _GM_THREAD_H_

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <thread>

struct SynchroObjects {
    static std::mutex synchro_mutex;
    std::mutex mutex;
    std::condition_variable cv;
};

static std::unordered_map<const void*, SynchroObjects> _synchro_objects;


// synchronized(container, [&]{
//     // stuff
// });
template <typename T, typename F>
void synchronized(T const& obj, F&& func)
{
    std::lock_guard<std::mutex> outer(SynchroObjects::synchro_mutex);
    std::lock_guard<std::mutex> inner(_synchro_objects[&obj].mutex);
    std::forward<F>(func)();
}

template <typename T>
SynchroObjects& getSynchro(T const& container) {
    std::lock_guard<std::mutex> lk(SynchroObjects::synchro_mutex);
    return _synchro_objects[&container];
}

template <typename T>
void notify(T const& container) {
    getSynchro(container).cv.notify_one();
}

template <typename T>
void notifyAll(T const& container) {
    getSynchro(container).cv.notify_all();
}

template <typename T>
void wait(T const& container) {
    // potentially you could add some safety checking to ensure
    // that _synchro_objects[&container] exists here, maybe use
    // at() in debug mode or something

    auto& sync = getSynchro(container);
    std::unique_lock<std::mutex> lk(sync.mutex);
    sync.cv.wait(lk);
}

namespace gm {

class Runnable {
public:
    void setRunning(bool nrun) {running = nrun;}
    bool isRunning() {return running;}
#ifndef  __CUDA_ARCH__
    Runnable() : callback(NULL), run_function([&]{this->run();}), running(false) {}
    Runnable(Runnable* nrun) : callback(nrun->callback), run_function(nrun->run_function), running(false) {}
    Runnable(std::function <void()>myrun) : run_function(myrun), running(false) {}
    Runnable(std::function <void()>myrun, std::function <void (Runnable* mycall)>mycallback) : 
    run_function(myrun),
    callback(mycallback), running(false) {
    }
    ~Runnable() {running = false;}
    //Lambda setCallback
    template<typename Func>
    void setCallback (Func func) {
        callback = func;
    }
    template<typename Func>
    void setRun (Func func) {
        run_function = func;
    }
    //Standard setCallback
    //void (*callback)(Runnable* mycall);
    //void setCallback(void(*setcall)(Runnable* mycall)) {callback = setcall;}
    virtual void doCallback(Runnable* myrun) {
        if (callback) callback(myrun);
    }
    virtual void run() {
        if (run_function) {
            //running = true;
            run_function();
            //running = false;
        }
    }
private:
    std::function <void (Runnable* mycall)>callback;
    std::function <void()>run_function;
#endif //__CUDA_ARCH__
    bool running;
};

class Thread : public Runnable {
private:
    std::thread thd;
    static void* startThread(void* thd);
    Runnable* myrunnable;
public:
    Thread() : Runnable(), myrunnable(this) {}
#ifndef  __CUDA_ARCH__
    Thread(Runnable* runnable) : Runnable(runnable), myrunnable(runnable) {}
    Thread(Runnable &runnable) : Runnable(runnable), myrunnable(&runnable) {}
    Thread(std::function <void()>myrun) : Runnable(myrun), myrunnable(this) {}
    Thread(std::function <void()>myrun, std::function <void (Runnable* mycall)>mycallback) : 
        Runnable(myrun, mycallback), myrunnable(this) {}
#endif //__CUDA_ARCH__
    ~Thread(){setRunning(false);}
    void start();
    void join();
};

}

#endif //_GM_THREAD_H_
