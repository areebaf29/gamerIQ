function buildNav(activePage) {
  const pages = [
    {id:'predict', label:'Predict', href:'/predict.html'},
    {id:'history', label:'History', href:'/history.html'},
    {id:'models',  label:'ML Models', href:'/models.html'},
  ];
  return `<nav style="background:#1e1e2e;padding:0 2rem;display:flex;align-items:center;justify-content:space-between;height:56px;position:sticky;top:0;z-index:100;border-bottom:1px solid #2a2a4a;">
    <span style="color:#a78bfa;font-weight:700;font-size:17px;letter-spacing:-0.3px;">🎮 GamerIQ</span>
    <div style="display:flex;gap:4px;">
      ${pages.map(p=>`<a href="${p.href}" style="padding:6px 16px;border-radius:6px;font-size:13px;font-weight:500;text-decoration:none;color:${p.id===activePage?'white':'#888'};background:${p.id===activePage?'#a78bfa':'transparent'}">${p.label}</a>`).join('')}
    </div>
  </nav>`;
}
