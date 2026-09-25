/* Shared authenticated navigation + small helpers. */

const PLATFORMS = ['codeforces', 'atcoder', 'leetcode'];
const PLATFORM_LABEL = {codeforces: 'Codeforces', atcoder: 'AtCoder', leetcode: 'LeetCode'};

function initials(user) {
    const f = (user.first_name || '').trim();
    const l = (user.last_name || '').trim();
    if (f || l) return ((f[0] || '') + (l[0] || '')).toUpperCase();
    return (user.username || '?').slice(0, 2).toUpperCase();
}

function avatarHtml(user, cls) {
    const c = 'avatar ' + (cls || '');
    return user.avatar
        ? `<img src="${user.avatar}" class="${c}" alt="">`
        : `<div class="${c} avatar-placeholder">${initials(user)}</div>`;
}

/**
 * Render the top navigation for signed-in pages.
 * Redirects to /login if there is no session.
 */
async function renderNav(active) {
    let user;
    try {
        user = await (await fetch('/api/me')).json();
    } catch {
        location.href = '/login';
        return null;
    }
    if (!user.logged_in) {
        location.href = '/login?next=' + encodeURIComponent(location.pathname);
        return null;
    }

    const link = (href, label) =>
        `<a href="${href}" class="${active === href ? 'active' : ''}">${label}</a>`;

    document.getElementById('nav').innerHTML = `
      <div class="nav-inner">
        <a href="/app" class="brand">C<span>PRS</span></a>
        <div class="nav-links">
          ${link('/app', 'Recommendations')}
          ${link('/report', 'My report')}
          ${link('/contests', 'Contests')}
          ${link('/profile', 'Profile')}
        </div>
        <div class="nav-right">
          <div class="user-menu" id="userMenu">
            <button class="user-btn" onclick="toggleMenu(event)">
              ${avatarHtml(user, 'avatar-sm')}
              <span>${user.first_name || user.username}</span>
              <span style="color:var(--text-dim);font-size:0.7rem">▾</span>
            </button>
            <div class="menu-drop" id="menuDrop">
              <div class="menu-head">
                <div style="font-weight:600">${user.username}</div>
                <div style="color:var(--text-dim);font-size:0.8rem">${user.email || 'No email set'}</div>
              </div>
              <a href="/profile">Profile &amp; accounts</a>
              <a href="/report">My report</a>
              <a href="#" onclick="doLogout(event)">Sign out</a>
            </div>
          </div>
        </div>
      </div>`;

    document.addEventListener('click', () => {
        const d = document.getElementById('menuDrop');
        if (d) d.classList.remove('open');
    });
    return user;
}

function toggleMenu(e) {
    e.stopPropagation();
    document.getElementById('menuDrop').classList.toggle('open');
}

async function doLogout(e) {
    if (e) e.preventDefault();
    await fetch('/api/logout', {method: 'POST'});
    location.href = '/';
}

/** Colour a 0..1 rate as a bar class. */
function rateClass(rate) {
    if (rate >= 0.66) return 'good';
    if (rate >= 0.4) return 'mid';
    return 'low';
}

function bar(rate) {
    const pct = Math.round((rate || 0) * 100);
    return `<div class="bar-track"><div class="bar-fill ${rateClass(rate)}" style="width:${pct}%"></div></div>`;
}

function showAlert(id, kind, msg) {
    const el = document.getElementById(id);
    el.className = 'alert alert-' + kind + ' show';
    el.textContent = msg;
}
function hideAlert(id) {
    const el = document.getElementById(id);
    if (el) el.className = 'alert';
}
