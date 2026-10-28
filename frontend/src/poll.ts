// Following a run live, without React: when a poll of its version reads the run again, where the page's clock stands,
// and the pause. Times are milliseconds; `now` is the browser's time, `server` the server's at the last read.

export type Follow = {
  version: string // the version the run was last read at
  at: number // the browser's time of that read
  server: number // the server's time of that read
  paused: number | null // the browser's time of the pause
}

export const follow = (): Follow => ({ version: '', at: 0, server: 0, paused: null })

// The run was read at `version`, the server's time then being `server`.
export const read = (f: Follow, version: string, server: number, now: number): Follow => ({ ...f, version, at: now, server })

// Read again at least this often while the run has not stopped: a stall or a crash changes no file of the run.
const REREAD_MS = 60_000

// What a poll that found the run at `version` does: read the run again, nothing as the run is the same, or, paused,
// only say that the run changed since it was last read.
export function polled(f: Follow, version: string, now: number, stopped: boolean): 'read' | 'same' | 'changed' {
  if (f.paused != null) return version !== f.version ? 'changed' : 'same'
  if (version !== f.version) return 'read'
  return !stopped && now - f.at > REREAD_MS ? 'read' : 'same'
}

export const pause = (f: Follow, now: number): Follow => ({ ...f, paused: now })

// Resumed, the next poll reads the run whatever its version.
export const resume = (f: Follow): Follow => ({ ...f, paused: null, version: '' })

// The page's clock: the server's time of the last read moved on by the browser's time since, or till the pause.
export const clock = (f: Follow, now: number): number => f.server + ((f.paused ?? now) - f.at)
