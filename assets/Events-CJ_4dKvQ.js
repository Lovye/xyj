import{_ as f,c as o,a,w as n,b as e,F as d,r as _,d as r,o as s,e as g,t as c}from"./index-DX8bgldV.js";import{L as x,R as y}from"./RightCircleOutlined-B8NVBWoB.js";const i="/xyj/assets/event-MUswaZo6.png",w={name:"Events",components:{LeftCircleOutlined:x,RightCircleOutlined:y},data(){return{carouselImages:[i,i,i],events:[]}},created(){this.fetchEvents()},methods:{async fetchEvents(){this.events=[{id:1,name:"创意市集",initiator:"艺术设计协会",description:"创意市集旨在为学生提供一个展示才艺和创意的平台。学生可以通过摆摊售卖自己设计的手工作品、绘画、摄影作品，甚至是手工食品。活动还设有互动区，如DIY手工坊、创意比赛，让参与者体验艺术的魅力。",url:"event/1"},{id:2,name:"夜跑挑战赛",initiator:"校体育委员会",description:"夜跑挑战赛是一项倡导健康与环保的运动活动，路线围绕校园主要道路，灯光和音乐伴随整个赛道。参与者可选择3公里或5公里赛程，完成比赛的学生可获得纪念奖牌和环保奖品。活动旨在增强学生的身体素质，同时增进同学间的友谊。",url:"event/2"},{id:3,name:"百年名著共读会",initiator:"校文学社",description:"每月挑选一部世界文学名著，由专业导师带领学生进行共读与讨论。通过分享阅读心得、角色扮演、主题演讲等方式，帮助学生加深对名著的理解，并提升文学素养。本次活动还邀请校内外知名文学讲师做专题讲座。",url:"event/3"}]}}},k={class:"event-page"},E={class:"custom-slick-arrow",style:{left:"10px","z-index":"1"}},I={class:"custom-slick-arrow",style:{right:"10px"}},B=["src"],C={class:"event-list",style:{"margin-top":"24px"}},b=["href"],L={class:"event-header"},A={class:"event-name"},D={class:"event-initiator"},F={class:"event-description"};function N(O,R,V,$,l,j){const p=r("left-circle-outlined"),u=r("right-circle-outlined"),m=r("a-carousel"),h=r("a-card");return s(),o("div",k,[a(m,{autoplay:"",arrows:""},{prevArrow:n(()=>[e("div",E,[a(p)])]),nextArrow:n(()=>[e("div",I,[a(u)])]),default:n(()=>[(s(!0),o(d,null,_(l.carouselImages,(t,v)=>(s(),o("div",{key:v},[e("img",{src:t,alt:"活动轮播图",class:"carousel-image"},null,8,B)]))),128))]),_:1}),e("div",C,[(s(!0),o(d,null,_(l.events,t=>(s(),g(h,{key:t.id,class:"event-card",bordered:"",hoverable:""},{default:n(()=>[e("a",{href:t.url},[e("div",L,[e("h3",A,c(t.name),1),e("span",D,"发起人: "+c(t.initiator),1)]),e("p",F,c(t.description),1)],8,b)]),_:2},1024))),128))])])}const S=f(w,[["render",N],["__scopeId","data-v-71f13c50"]]);export{S as default};
