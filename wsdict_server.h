#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/* Start the wsdict WebSocket dictionary server in a background thread.
 * Blocks until the server is bound and listening on the given port.
 * static_dir: path to static files directory (NULL → "static").
 * state_file: path to persist uhd:config / uhd:channels across restarts (NULL → no persistence).
 * Returns 0 on success, -1 on failure.
 */
int wsdict_server_start(unsigned short port, const char* static_dir, const char* state_file);

/* Signal the server to shut down and wait for it to exit. */
void wsdict_server_stop(void);

#ifdef __cplusplus
}
#endif
