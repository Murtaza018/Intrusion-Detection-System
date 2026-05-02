importScripts(
  "https://www.gstatic.com/firebasejs/9.0.0/firebase-app-compat.js",
);
importScripts(
  "https://www.gstatic.com/firebasejs/9.0.0/firebase-messaging-compat.js",
);

firebase.initializeApp({
  apiKey: "AIzaSyC2LzNye1pdT9le2BdNXtOWR4o2R4lxj3Y",
  authDomain: "ids-fyp-53e2b.firebaseapp.com",
  projectId: "ids-fyp-53e2b",
  storageBucket: "ids-fyp-53e2b.firebasestorage.app",
  messagingSenderId: "981976698105",
  appId: "1:981976698105:web:3cc79e61fe54f3db62e628",
});

const messaging = firebase.messaging();

// Handle background messages
messaging.onBackgroundMessage((payload) => {
  console.log(
    "[firebase-messaging-sw.js] Received background message ",
    payload,
  );
  const notificationTitle = payload.notification.title;
  const notificationOptions = {
    body: payload.notification.body,
    icon: "/firebase-logo.png",
  };

  self.registration.showNotification(notificationTitle, notificationOptions);
});
