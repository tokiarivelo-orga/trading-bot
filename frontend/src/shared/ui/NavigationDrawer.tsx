"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect, createContext, useContext } from "react";
import { useAccounts } from "@/shared/api/account-context";

// Context for managing the drawer state globally
const NavigationContext = createContext<{
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}>({
  isOpen: false,
  setIsOpen: () => {},
});

export function NavigationProvider({ children }: { children: React.ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <NavigationContext.Provider value={{ isOpen, setIsOpen }}>
      {children}
    </NavigationContext.Provider>
  );
}

export function useNavigationDrawer() {
  return useContext(NavigationContext);
}

// Hamburger menu toggle button, paired with the account switcher — every
// page's header renders just `<MenuButton />`, so bundling the switcher here
// (rather than a separate component each page would need to add) puts
// account selection in the top nav everywhere with a single-file change.
export function MenuButton() {
  const { setIsOpen } = useNavigationDrawer();
  return (
    <div className="flex items-center gap-2">
      <button
        onClick={() => setIsOpen(true)}
        className="cursor-pointer rounded p-1.5 hover:bg-line text-ink-muted hover:text-ink transition-colors focus:outline-none"
        title="Open Menu"
        type="button"
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
      <AccountSwitcher />
    </div>
  );
}

/** Active-account dropdown — reads/writes `AccountContext` (see
 * `shared/api/account-context.tsx`). Renders nothing until the account list
 * loads, and nothing at all if only one account is configured (today's
 * single-account setups see no UI change). */
function AccountSwitcher() {
  const { accounts, activeAccountId, setActiveAccountId, loading } = useAccounts();

  if (loading || accounts.length === 0) return null;
  if (accounts.length === 1) {
    return (
      <span
        className="rounded border border-line px-2 py-1 text-xs text-ink-muted"
        title="Only one account is configured (configs/accounts.yaml)"
      >
        {accounts[0].label}
      </span>
    );
  }

  return (
    <select
      className="cursor-pointer rounded border border-line bg-panel px-2 py-1 text-xs text-ink focus:border-accent focus:outline-none"
      value={activeAccountId ?? ""}
      onChange={(e) => setActiveAccountId(e.target.value)}
      title="Active MT5 account"
    >
      {accounts.map((a) => (
        <option key={a.id} value={a.id}>
          {a.label} ({a.mode})
        </option>
      ))}
    </select>
  );
}

// Drawer Component
export function NavigationDrawer() {
  const { isOpen, setIsOpen } = useNavigationDrawer();
  const pathname = usePathname();

  // Automatically close drawer on route change
  useEffect(() => {
    setIsOpen(false);
  }, [pathname, setIsOpen]);

  // Handle ESC key press to close drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [setIsOpen]);

  const navItems = [
    {
      name: "Terminal (Chart)",
      path: "/",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
      ),
    },
    {
      name: "Bots",
      path: "/bots",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
    },
    {
      name: "Bot Control",
      path: "/bot-control",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v6m6.364 1.636a9 9 0 11-12.728 0" />
        </svg>
      ),
    },
    {
      name: "Backtests",
      path: "/backtest",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
        </svg>
      ),
    },
    {
      name: "History",
      path: "/history",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
    {
      name: "Analytics",
      path: "/analytics",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 17l6-6 4 4 8-8m0 0h-5m5 0v5" />
        </svg>
      ),
    },
    {
      name: "Activity log",
      path: "/logs",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h10M4 18h16" />
        </svg>
      ),
    },
    {
      name: "Indicators",
      path: "/indicators",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 17l4-8 4 4 4-10 6 14M3 20h18" />
        </svg>
      ),
    },
    {
      name: "AI Reports",
      path: "/ai-reports",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
        </svg>
      ),
    },
    {
      name: "News",
      path: "/news",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1M19 20a2 2 0 002-2V8a2 2 0 00-2-2h-2m0 0V4a2 2 0 00-2-2h-3m0 0H8M5.5 8h.01M5.5 12h.01M9 8h6M9 12h6" />
        </svg>
      ),
    },
    {
      name: "Optimizer & AI",
      path: "/optimizer",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
    },
    {
      name: "Model Dashboard",
      path: "/model-dashboard",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 002 2h2a2 2 0 002-2z" />
        </svg>
      ),
    },
    {
      name: "Settings",
      path: "/settings",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
    {
      name: "Documentation",
      path: "/docs",
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
    },
  ];

  // Helper to determine if link is active. Highlight root correctly, and highlight parent routes for sub-routes
  const isLinkActive = (path: string) => {
    if (path === "/") {
      return pathname === "/";
    }
    // /strategies/drafts/[id] and /strategies/versions/[id] are still real
    // routes (linked from the Bots hub) even though the nav entry points at
    // /bots — keep the drawer highlighting "Bots" while on either.
    if (path === "/bots" && pathname.startsWith("/strategies")) {
      return true;
    }
    return pathname === path || pathname.startsWith(path + "/");
  };

  return (
    <>
      {/* Backdrop with transition */}
      <div
        className={`fixed inset-0 z-40 bg-black/60 backdrop-blur-xs transition-opacity duration-300 ${
          isOpen ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={() => setIsOpen(false)}
      />

      {/* Drawer panel with slide transition */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex w-72 flex-col border-r border-line bg-panel/95 backdrop-blur-md shadow-2xl transition-transform duration-300 ease-in-out ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-line px-5 py-4">
          <div className="flex flex-col">
            <span className="text-base font-bold tracking-wide text-ink">AI Trading Bot</span>
            <span className="text-2xs text-ink-muted mt-0.5">Navigation Control</span>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="cursor-pointer rounded p-1 hover:bg-line text-ink-muted hover:text-ink transition-colors focus:outline-none"
            aria-label="Close menu"
            type="button"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Navigation list */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const active = isLinkActive(item.path);
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center gap-3.5 px-4 py-3 text-sm font-medium rounded-md transition-all duration-200 ${
                  active
                    ? "bg-accent/15 text-accent border-l-4 border-accent -ml-3 pl-3.5"
                    : "text-ink hover:bg-line/40 hover:text-ink border-l-4 border-transparent"
                }`}
              >
                <span className={active ? "text-accent" : "text-ink-muted hover:text-ink"}>
                  {item.icon}
                </span>
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-line p-4 text-center">
          <div className="flex items-center justify-center gap-1.5 text-2xs text-ink-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-ok animate-pulse" />
            <span>Trading Engine Connected</span>
          </div>
        </div>
      </aside>
    </>
  );
}
