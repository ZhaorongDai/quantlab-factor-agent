// A run read live: its version is polled every 5 s and the run read again only when it changed (or, while it has
// not stopped, at least every minute, so a stall or a crash shows although no file of the run changes). The clock is
// the server's time, moving on between two reads, so that "how long ago" does not depend on the browser's clock.
// Paused, the clock stands and the run is not read again; the version is still polled, only to say that the run
// changed since. Resuming reads the run at once. The rules are in poll.ts.
import { useEffect, useRef, useState } from 'react'
import * as api from './api'
import type { Run } from './api'
import * as poll from './poll'

const POLL_MS = 5_000

export type LiveRun = {
  run: Run | null
  error: string
  clock: number
  paused: number | null // the browser's time of the pause
  changed: boolean // paused, whether the run changed since it was last read
  pause: () => void
  resume: () => void
}

export function useLiveRun(name: string): LiveRun {
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState('')
  const [tick, setTick] = useState(Date.now())
  const [paused, setPaused] = useState<number | null>(null)
  const [changed, setChanged] = useState(false)
  const followed = useRef(poll.follow())
  const check = useRef<() => void>(() => {})
  // The latest run, for the poll's closure.
  const runRef = useRef<Run | null>(null)
  runRef.current = run

  useEffect(() => {
    let live = true
    setRun(null)
    setError('')
    setPaused(null)
    setChanged(false)
    followed.current = poll.follow()
    const once = async () => {
      try {
        const version = await api.version(name)
        if (!live) return
        const next = poll.polled(followed.current, version, Date.now(), !!runRef.current?.summary.stop)
        if (next === 'changed') setChanged(true)
        if (next === 'read') {
          const read = await api.run(name)
          // Paused while the run was being read: the page stays as it stood, and the next poll says it changed.
          if (!live || followed.current.paused != null) return
          followed.current = poll.read(followed.current, version, Date.parse(read.now), Date.now())
          setRun(read)
          setChanged(false)
        }
        setError('')
      } catch (e) {
        if (live) setError(String(e))
      }
      if (live) setTick(Date.now())
    }
    check.current = once
    once()
    const timer = setInterval(once, POLL_MS)
    return () => { live = false; clearInterval(timer) }
  }, [name])

  const pause = () => {
    const now = Date.now()
    followed.current = poll.pause(followed.current, now)
    setPaused(now)
  }
  const resume = () => {
    followed.current = poll.resume(followed.current)
    setPaused(null)
    check.current()
  }
  const clock = run ? poll.clock(followed.current, tick) : tick
  return { run, error, clock, paused, changed, pause, resume }
}

// The tab's title: the run's name, the factors accepted since the page opened and whether the run stopped.
export function useTabTitle(name: string, run: Run | null) {
  const first = useRef<{ name: string; accepted: number } | null>(null)
  useEffect(() => {
    if (!run) return
    if (first.current?.name !== name) first.current = { name, accepted: run.summary.accepted }
    const fresh = run.summary.accepted - first.current.accepted
    document.title = `${fresh > 0 ? `(${fresh} new) ` : ''}${run.summary.stop ? '[stopped] ' : ''}${name} · factor-agent`
  }, [name, run])
}
