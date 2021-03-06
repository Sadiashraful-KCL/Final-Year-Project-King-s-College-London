//
//  AppTutorialViewController.swift
//  CalorieTracker.ai
//
//  Created by Sadi Ashraful on 26/11/2018.
//  Copyright © 2018 Sadi Ashraful. All rights reserved.
//

import UIKit
import paper_onboarding

class AppTutorialViewController: UIViewController, PaperOnboardingDataSource, PaperOnboardingDelegate {
    
    @IBOutlet weak var getStartedButton: UIButton!
    
    func onboardingItemsCount() -> Int {
        return 2
    }
    
    func onboardingItem(at index: Int) -> OnboardingItemInfo {
        return [
            OnboardingItemInfo(informationImage: #imageLiteral(resourceName: "AppLogo2"),
                               title: "Welcome to CalorieTracker.ai",
                               description: "Please take a few minutes to learn the features of the app",
                               pageIcon: #imageLiteral(resourceName: "AppLogo2"),
                               color: #colorLiteral(red: 0.4392156899, green: 0.01176470611, blue: 0.1921568662, alpha: 1),
                               titleColor: UIColor.white,
                               descriptionColor: UIColor.white,
                               titleFont: UIFont.init(name: "AvenirNext-Bold", size: 24)!,
                               descriptionFont: UIFont.init(name: "AvenirNext-Regular", size: 18)!),
            
            OnboardingItemInfo(informationImage: #imageLiteral(resourceName: "iconfinder_Camera_2998131"),
                               title: "Tracker",
                               description: "Take a picture or choose from your albums and let the app track down the calories",
                               pageIcon: #imageLiteral(resourceName: "iconfinder_Camera_2998131"),
                               color: #colorLiteral(red: 0.2549019754, green: 0.2745098174, blue: 0.3019607961, alpha: 1),
                               titleColor: UIColor.white,
                               descriptionColor: UIColor.white,
                               titleFont: UIFont.init(name: "AvenirNext-Bold", size: 24)!,
                               descriptionFont: UIFont.init(name: "AvenirNext-Regular", size: 18)!)
            ][index]
    }
    
    
   

    internal func onboardingDidTransitonToIndex(_ index: Int) {
        if index == 1 {
//            UIView.animate(withDuration: 0.2, animations: {
//                self.getStartedButton.alpha = 1
//            })
            UIView.animate(withDuration: 0.2, animations: {
                self.getStartedButton.alpha = 1
            })
        }
    }

    
     @IBOutlet weak var onboardingView: OnboardingView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        onboardingView.dataSource = self
        onboardingView.delegate = self
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    

   

}
