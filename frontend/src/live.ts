// A run read live: its version is polled every 5 s and the run read again only when it changed (or, while it has
// not stopped, at least every minute, so a stall or a crash shows although no file of the run changes). The clock is
// the server's time, moving on between two reads, so that "how long ago" does not depend on the browser's clock.
import { useEffect, useRef, useState } from 'react'
import * as api from './api'
import type { Run } from './api'

const POLL_MS = 5_000
const REREAD_MS = 60_000

export function useLiveRun(name: string): { run: Run | null; error: string; clock: number } {
  const [run, setRun] = useState<Run | null>(null)
  const [error, setError] = useState('')
  const [tick, setTick] = useState(Date.now())
  const read = useRef({ version: '', at: 0 })
  // The latest run, for the poll's closure.
  const runRef = useRef<Run | null>(null)
  runRef.current = run

  useEffect(() => {
    let live = true
    setRun(null)
    setError('')
    read.current = { version: '', at: 0 }
    const reread = async (version: string) => {
      const next = await api.run(name)
      if (!live) return
      read.current = { version, at: Date.now() }
      setRun(next)
    }
    const poll = async () => {
      try {
        const version = await api.version(name)
        if (!live) return
        const stale = Date.now() - read.current.at > REREAD_MS
        if (version !== read.current.version || (stale && !runRef.current?.summary.stop)) await reread(version)
        setError('')
      } catch (e) {
        if (live) setError(String(e))
      }
      if (live) setTick(Date.now())
    }
    poll()
    const timer = setInterval(poll, POLL_MS)
    return () => { live = false; clearInterval(timer) }
  }, [name])

  const clock = run ? Date.parse(run.now) + (tick - read.current.at) : tick
  return { run, error, clock }
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
