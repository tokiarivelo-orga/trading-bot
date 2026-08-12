"use client";

import { useState, useTransition, useEffect } from "react";
import { Power, PowerOff } from "lucide-react";
import { toggleBotState, checkBotState } from "./actions";

export default function ToggleBotButton({ symbol }: { symbol: string }) {
  const [isPending, startTransition] = useTransition();
  const [isActive, setIsActive] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  useEffect(() => {
    checkBotState(symbol).then((res) => setIsActive(res.active));
  }, [symbol]);

  const handleToggle = () => {
    startTransition(async () => {
      setStatusMsg(null);
      const res = await toggleBotState(symbol, !isActive);
      if (res.success) {
        setIsActive(!isActive);
      }
      setStatusMsg(res.message);
      setTimeout(() => setStatusMsg(null), 5000);
    });
  };

  return (
    <div className="flex items-center gap-3">
      {statusMsg && (
        <span className={`text-xs ${statusMsg.includes("Erreur") ? "text-error" : "text-ok"} animate-pulse whitespace-nowrap`}>
          {statusMsg}
        </span>
      )}
      
      <button
        onClick={handleToggle}
        disabled={isPending}
        className={`flex items-center gap-2 px-4 py-2.5 h-[42px] font-semibold rounded-lg border transition-all shadow-[0_0_15px_rgba(var(--accent-rgb),0.15)] disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap ${
          isActive 
            ? "bg-error/10 hover:bg-error/20 text-error border-error/30" 
            : "bg-ok/10 hover:bg-ok/20 text-ok border-ok/30"
        }`}
      >
        {isActive ? (
          <><PowerOff size={16} className={isPending ? "animate-pulse" : ""} /> {isPending ? "..." : "Désactiver Bot"}</>
        ) : (
          <><Power size={16} className={isPending ? "animate-pulse" : ""} /> {isPending ? "..." : "Activer Bot"}</>
        )}
      </button>
    </div>
  );
}
