(function () {
  const paths = {
    squarePen: '<path d="M12 20H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path><path d="m16 5 3 3"></path><path d="m10 14 8.5-8.5a2.12 2.12 0 0 1 3 3L13 17l-4 1 1-4Z"></path>',
    search: '<circle cx="11" cy="11" r="7"></circle><path d="m20 20-4-4"></path>',
    folder: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2h7A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z"></path>',
    folderPlus: '<path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4l2 2h7A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z"></path><path d="M12 11v5m-2.5-2.5h5"></path>',
    plus: '<path d="M12 5v14m-7-7h14"></path>',
    moreHorizontal: '<circle cx="5" cy="12" r="1"></circle><circle cx="12" cy="12" r="1"></circle><circle cx="19" cy="12" r="1"></circle>',
    pencil: '<path d="M4 20h4L19 9a2.1 2.1 0 0 0-3-3L5 17v3Z"></path><path d="m14.5 7.5 2 2"></path>',
    pin: '<path d="m16 3 5 5-4 1-4 4 1 4-2 2-3-6-6-3 2-2 4 1 4-4z"></path><path d="m9 15-5 5"></path>',
    trash2: '<path d="M3 6h18m-2 0-.9 14H5.9L5 6m4 0V4h6v2m-5 4v6m4-6v6"></path>',
    send: '<path d="m22 2-7 20-4-9-9-4Z"></path><path d="M22 2 11 13"></path>',
    square: '<rect x="5" y="5" width="14" height="14" rx="2"></rect>',
    paperclip: '<path d="m21.4 11.1-8.5 8.5a6 6 0 0 1-8.5-8.5l8.5-8.5a4 4 0 0 1 5.7 5.7l-8.5 8.5a2 2 0 0 1-2.8-2.8l8.5-8.5"></path>',
    userCircle: '<circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="8" r="3"></circle><path d="M5.5 20a7 7 0 0 1 13 0"></path>',
    x: '<path d="m18 6-12 12M6 6l12 12"></path>',
    arrowLeft: '<path d="m12 19-7-7 7-7m-7 7h14"></path>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><path d="m7 10 5 5 5-5m-5 5V3"></path>',
    copy: '<rect x="8" y="8" width="11" height="11" rx="2"></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path>',
    menu: '<path d="M4 6h16M4 12h16M4 18h16"></path>',
    chevronDown: '<path d="m6 9 6 6 6-6"></path>',
    chevronRight: '<path d="m9 18 6-6-6-6"></path>',
    sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"></path>',
    moon: '<path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"></path>',
    graduationCap: '<path d="m2 10 10-5 10 5-10 5-10-5Z"></path><path d="M6 12v5c3.3 2.7 8.7 2.7 12 0v-5m4-2v6"></path>',
    sparkles: '<path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-1.9-5.8L4 11l6.1-2.2L12 3Z"></path><path d="m19 14 1.2 2.8L23 18l-2.8 1.2L19 22l-1.2-2.8L15 18l2.8-1.2L19 14ZM5 2l.8 1.8L8 5l-2.2 1.2L5 8l-.8-1.8L2 5l2.2-1.2L5 2Z"></path>',
    logOut: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><path d="m16 17 5-5-5-5m5 5H9"></path>',
    logIn: '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"></path><path d="m10 17 5-5-5-5m5 5H3"></path>',
    userPlus: '<path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="10" cy="7" r="4"></circle><path d="M19 8v6m-3-3h6"></path>',
    check: '<path d="m5 12 4 4L19 6"></path>',
    arrowDown: '<path d="M12 5v14m7-7-7 7-7-7"></path>',
    retry: '<path d="M20 11a8 8 0 0 0-14.9-4L3 10"></path><path d="M3 5v5h5m-4 3a8 8 0 0 0 14.9 4L21 14"></path><path d="M21 19v-5h-5"></path>',
    pdf: '<path d="M6 3h8l4 4v14H6z"></path><path d="M14 3v5h5m-10 8h7m-7-3.5h3"></path>',
    youtube: '<rect x="3" y="6" width="18" height="12" rx="3"></rect><path d="m10 9 5 3-5 3z" fill="currentColor" stroke="none"></path>',
    sources: '<path d="M5 4.5h9a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3V4.5Z"></path><path d="M8 4.5v11.8c0 2 1.2 3.7 3.2 3.7m-2.7-12h5.5m-5.5 3.5h5.5"></path>'
  };

  window.leeLearnIcon = function (name) {
    const content = paths[name];
    if (!content) {
      throw new Error(`Unknown LeeLearn icon: ${name}`);
    }
    return `<svg class="ui-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${content}</svg>`;
  };

  document.querySelectorAll('[data-icon]').forEach((slot) => {
    slot.innerHTML = window.leeLearnIcon(slot.dataset.icon);
  });
})();
