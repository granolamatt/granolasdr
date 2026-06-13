#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Start the TCI WebSocket server on `port` with INT16 gain `gain`.
 * Requires wsdict_server_start() to have been called first.
 * Blocks until bound and listening. Returns 0 on success, -1 on failure.
 * Pass tci_port=0 to disable TCI entirely. */
int tci_server_start(uint16_t port, float gain);

/* Abort the TCI server task. */
void tci_server_stop(void);

/* Push `count` float32 PCM samples for sink `trx` into the audio accumulator.
 * Broadcasts a binary TCI frame when 480 samples have accumulated (10 ms at 48 kHz).
 * Non-blocking; slow clients drop frames rather than stalling the caller. */
void tci_push_audio(uint32_t trx, const float* pcm, uint32_t count);

/* Broadcast S-meter level `dbfs` (dBFS) for sink `trx` to all TCI clients.
 * Call at ~1 Hz from audioWorker using norm_ema_h_[composite_bin] as source. */
void tci_push_smeter(uint32_t trx, float dbfs);

/* Drain one VFO retune command from the Rust queue into *trx_out and *freq_out.
 * Returns 1 if a command was available, 0 if the queue is empty.
 * Call in a loop to drain all pending commands. */
int tci_poll_vfo(uint32_t* trx_out, uint64_t* freq_out);

#ifdef __cplusplus
}
#endif
