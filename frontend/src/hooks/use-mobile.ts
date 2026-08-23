import * as React from "react"

const MOBILE_BREAKPOINT = 768

export function useIsMobile() {
  const [isMobile, setIsMobile] = React.useState<boolean | undefined>(undefined)

  React.useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`)
    if (!mql) return

    const onChange = () => {
      setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    }
    if (mql.addEventListener) {
      mql.addEventListener("change", onChange)
    } else if ((mql as any).addListener) {
      (mql as any).addListener(onChange)
    }
    setIsMobile(window.innerWidth < MOBILE_BREAKPOINT)
    return () => {
      if (mql.removeEventListener) {
        mql.removeEventListener("change", onChange)
      } else if ((mql as any).removeListener) {
        (mql as any).removeListener(onChange)
      }
    }
  }, [])

  return !!isMobile
}
